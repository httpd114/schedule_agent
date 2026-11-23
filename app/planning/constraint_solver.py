"""轻量约束求解器：基于 python-constraint 的时间窗 / 资源 / 制度约束校验。

- detect_conflicts：与既有日程的时间、会议室、参与人冲突（分类型）
- check_rules：只校验 hard_constraints，baseline 规则不进入这里
- evaluate_plan：加权打分（冲突权重、违规权重、里程碑延后惩罚）+ 代价自动映射
- suggest_windows：用 CSP 求 Top3 可行时间窗口 + 可用会议室，而不是只提示“撞了”
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from constraint import Problem

from app.planning.rules import BusinessRule
from app.utils.time_periods import PERIOD_BOUNDS
from app.utils.validators import parse_datetime

DEFAULT_ROOMS = ["12F-A301", "15F-B205", "10F-B102", "16F-C301", "8F-A101"]

# 权重表：不同类型风险代价不同，不再是简单计数
CONFLICT_WEIGHTS = {"room_busy": 10, "attendee_busy": 8, "time_overlap": 5}
VIOLATION_WEIGHTS = {"minutes_delay": 6, "approval_violation": 7, "room_format": 3, "other": 5}
MILESTONE_DELAY_PENALTY = 12
BASE_SCORE = 100


@dataclass
class TimeWindow:
    start: datetime
    end: datetime


def _attendee_set(value: Any) -> set[str]:
    if isinstance(value, list):
        return {str(item).strip() for item in value if str(item).strip()}
    if not value:
        return set()
    return {part.strip() for part in str(value).replace("，", ",").split(",") if part.strip()}


def plan_window(plan: dict, default_minutes: int = 60) -> TimeWindow | None:
    start_raw = plan.get("start_time")
    if not start_raw:
        return None
    start = parse_datetime(start_raw, "start_time")
    end = parse_datetime(plan["end_time"], "end_time") if plan.get("end_time") else start + timedelta(minutes=default_minutes)
    if end <= start:
        end = start + timedelta(minutes=default_minutes)
    return TimeWindow(start, end)


def detect_conflicts(
    window: TimeWindow,
    schedules: list[dict],
    *,
    location: str | None = None,
    attendees: Any = None,
    ignore_ids: set[int] | None = None,
) -> list[dict]:
    """检测时间重叠、会议室占用与参与人冲突，并标记冲突类型。"""
    conflicts: list[dict] = []
    wanted_people = _attendee_set(attendees)
    for item in schedules:
        if ignore_ids and item.get("id") in ignore_ids:
            continue
        try:
            item_start = parse_datetime(item["start_time"], "start_time")
        except Exception:  # noqa: BLE001 - 脏数据跳过，不能污染校验
            continue
        item_end = (
            parse_datetime(item["end_time"], "end_time")
            if item.get("end_time")
            else item_start + timedelta(hours=1)
        )
        if not (window.start < item_end and item_start < window.end):
            continue

        reasons = ["时间段重叠"]
        types = ["time_overlap"]
        if location and item.get("location") and str(item["location"]).strip() == str(location).strip():
            reasons.append("会议室已被占用")
            types.append("room_busy")
        overlap_people = wanted_people & _attendee_set(item.get("attendees"))
        if overlap_people:
            reasons.append(f"参与人时间冲突：{'、'.join(sorted(overlap_people))}")
            types.append("attendee_busy")
        conflicts.append(
            {
                "schedule_id": item.get("id"),
                "title": item.get("title"),
                "time": f"{item['start_time']} ~ {item.get('end_time') or '未设置'}",
                "reason": "；".join(reasons),
                "types": types,
            }
        )
    return conflicts


def _violation_kind(rule: BusinessRule) -> str:
    text = f"{rule.enforce} {rule.rule_id} {rule.content}"
    if rule.enforce == "minutes_deadline" or "纪要" in text:
        return "minutes_delay"
    if rule.enforce == "approval_flow" or "审批" in text or "审核" in text:
        return "approval_violation"
    if rule.enforce == "room_format" or "命名" in text or "格式" in text:
        return "room_format"
    return "other"


def check_rules(
    window: TimeWindow,
    rules: list[BusinessRule],
    *,
    location: str | None = None,
    now: datetime | None = None,
) -> list[dict]:
    """校验结构化制度**硬约束**。baseline 规则不会传入本函数。"""
    current = now or datetime.now()
    violations: list[dict] = []
    for rule in rules:
        if not rule.valid or rule.type != "constraint" or not rule.enforce or rule.scope != "hard":
            continue
        hit = False
        if rule.enforce == "min_lead_hours":
            hours = float(rule.params.get("hours", 2))
            hit = (window.start - current).total_seconds() / 3600 < hours
        elif rule.enforce == "min_notice_days":
            days = int(rule.params.get("days", 3))
            hit = (window.start.date() - current.date()).days < days
        elif rule.enforce in {"minutes_deadline", "expense_window"}:
            hours = int(rule.params.get("hours", 24))
            hit = (window.start - current).total_seconds() / 3600 < hours
        elif rule.enforce == "work_hours":
            work_start = rule.params.get("start", "09:00")
            work_end = rule.params.get("end", "18:00")
            start_h, start_m = (int(part) for part in work_start.split(":"))
            end_h, end_m = (int(part) for part in work_end.split(":"))
            lower = window.start.replace(hour=start_h, minute=start_m)
            upper = window.start.replace(hour=end_h, minute=end_m)
            hit = window.start < lower or window.end > upper
        elif rule.enforce == "room_format" and location:
            pattern = rule.params.get("pattern", r"^\d+[Ff]-")
            hit = re.match(pattern, str(location).strip()) is None
        if hit:
            kind = _violation_kind(rule)
            violations.append(
                {
                    "rule_id": rule.rule_id,
                    "content": rule.content,
                    "kind": kind,
                    "weight": VIOLATION_WEIGHTS[kind],
                }
            )
    return violations


def _cost_from_score(score: int) -> str:
    """代价标签由分数区间自动映射，不再依赖 LLM 硬编码。"""
    if score >= 80:
        return "low"
    if score >= 55:
        return "medium"
    return "high"


def evaluate_plan(
    plan: dict,
    context: dict,
    rules: list[BusinessRule],
    *,
    now: datetime | None = None,
) -> dict:
    """加权打分：total = base - conflict - violation - risk_penalty。"""
    current = now or datetime.now()
    window = plan_window(plan)
    conflicts: list[dict] = []
    violations: list[dict] = []
    location = None
    attendees = None
    creates_schedule = False
    for step in plan.get("steps", []):
        if not isinstance(step, dict):
            continue
        if step.get("tool") == "create_schedule":
            creates_schedule = True
        args = step.get("args", {}) or {}
        if args.get("room_prefer") or args.get("location"):
            location = args.get("room_prefer") or args.get("location")
        if args.get("attendees"):
            attendees = args["attendees"]
    # 方案里的时间若只是“说明现有会议时间”，并未新建日程，就不应算冲突或违规
    if window and creates_schedule:
        conflicts = detect_conflicts(
            window, context.get("schedules", []), location=location, attendees=attendees
        )
        # 只校验硬约束；常态化基线根本不会传进来
        violations = check_rules(window, rules, location=location, now=current)

    plan_text = " ".join(
        [str(plan.get("title", "")), str(plan.get("description", ""))]
        + [str(step.get("args", {})) for step in plan.get("steps", []) if isinstance(step, dict)]
    )

    conflict_score = 0
    for conflict in conflicts:
        for kind in set(conflict.get("types") or ["time_overlap"]):
            conflict_score += CONFLICT_WEIGHTS.get(kind, 5)
    violation_score = sum(item.get("weight", 5) for item in violations)

    # 只有方案本身确实在推迟/改期时，才对“临近里程碑的高优待办被搁置”计惩罚；
    # 普通新建方案不应被上下文里无关的紧急待办扣分。
    delay_hint = any(
        word in plan_text for word in ("延后", "推迟", "延期", "改期", "稍后再定", "缓一缓", "改天")
    )
    milestone_risks: list[dict] = []
    priority_loss = 0
    if delay_hint:
        # 延后风险针对全部未完成任务，必须看全局 pending_todos；
        # 且只读取 todo 自身的 due_time，绝不读关联日程时间。
        for todo in (context.get("pending_todos") or context.get("todos", [])):
            title = str(todo.get("title") or "")
            priority = str(todo.get("priority") or "")
            due = todo.get("due_time")
            near = False
            if due:
                try:
                    due_dt = parse_datetime(due, "due_time")
                    near = 0 <= (due_dt.date() - current.date()).days <= 3
                except Exception:  # noqa: BLE001
                    near = False
            is_key = priority == "high" or "里程碑" in title
            covered = bool(title) and title in plan_text
            if is_key and not covered:
                priority_loss += 1
                if near:
                    milestone_risks.append({"title": title, "due_time": due})
    risk_penalty = min(len(milestone_risks), 2) * MILESTONE_DELAY_PENALTY

    total = max(0, BASE_SCORE - conflict_score - violation_score - risk_penalty)
    return {
        "conflict_count": len(conflicts),
        "violation_count": len(violations),
        "priority_loss": priority_loss,
        "conflict_score": conflict_score,
        "violation_score": violation_score,
        "risk_penalty": risk_penalty,
        "total_score": total,
        "score": total,
        "base_score": BASE_SCORE,
        "cost": _cost_from_score(total),
        "feasible": not conflicts and not violations,
        "conflicts": conflicts,
        "violations": violations,
        "milestone_risks": milestone_risks,
        "weights": {"conflict": CONFLICT_WEIGHTS, "violation": VIOLATION_WEIGHTS, "milestone": MILESTONE_DELAY_PENALTY},
    }


def _room_pattern(rules: list[BusinessRule]) -> str | None:
    for rule in rules:
        if rule.enforce == "room_format" and rule.valid and rule.scope == "hard":
            return str(rule.params.get("pattern", r"^\d+[Ff]-"))
    return None


def suggest_windows(
    desired_start: str | datetime,
    duration_minutes: int,
    context: dict,
    rules: list[BusinessRule],
    *,
    top_k: int = 3,
    span_days: int = 5,
    step_minutes: int = 30,
    rooms: list[str] | None = None,
    now: datetime | None = None,
    period: str | None = None,
    prefer_room: str | None = None,
    only_day: bool = False,
) -> list[dict]:
    """用 CSP 枚举可行起点，返回 Top3【具体时段 + 可用会议室】。

    ug1：period 给定时只在「上午/下午/晚上」区间内枚举；only_day=True 时
    仅检索 desired 当天（模糊时段“改到27号下午”不应跑到别的日子）。
    prefer_room 为首选会议室，排在候选池最前，冲突时仍由求解器换房。
    """
    desired = (
        parse_datetime(desired_start, "desired_start")
        if isinstance(desired_start, str)
        else desired_start
    )
    room_pool = [str(room) for room in (rooms or DEFAULT_ROOMS) if str(room).strip()]
    if prefer_room:
        room_pool = [str(prefer_room)] + [room for room in room_pool if room != str(prefer_room)]
    pattern = _room_pattern(rules)
    if pattern:
        room_pool = [room for room in room_pool if re.match(pattern, room)] or room_pool

    # 时段区间：period 优先，否则全天工作时段
    if period:
        lower_h, upper_h = PERIOD_BOUNDS.get(period, (9, 18))
        minute_range = range(lower_h * 60, upper_h * 60 - duration_minutes + 1, step_minutes)
    else:
        minute_range = range(9 * 60, 18 * 60 - duration_minutes + 1, step_minutes)

    domain: list[datetime] = []
    day_offsets = (0,) if only_day else range(-1, span_days + 1)
    for day_offset in day_offsets:
        day = desired.date() + timedelta(days=day_offset)
        for minutes in minute_range:
            domain.append(datetime(day.year, day.month, day.day) + timedelta(minutes=minutes))
    if not domain:
        return []

    schedules = context.get("schedules", [])

    def room_at(start: datetime, end: datetime) -> str | None:
        occupied = set()
        for item in schedules:
            try:
                item_start = parse_datetime(item["start_time"], "start_time")
            except Exception:  # noqa: BLE001
                continue
            item_end = (
                parse_datetime(item["end_time"], "end_time")
                if item.get("end_time")
                else item_start + timedelta(hours=1)
            )
            if start < item_end and item_start < end and item.get("location"):
                occupied.add(str(item["location"]).strip())
        return next((room for room in room_pool if room not in occupied), None)

    def accepted(start: datetime) -> bool:
        window = TimeWindow(start, start + timedelta(minutes=duration_minutes))
        return (
            not detect_conflicts(window, schedules)
            and not check_rules(window, rules, now=now)
            and room_at(window.start, window.end) is not None
        )

    problem = Problem()
    problem.addVariable("start", domain)
    problem.addConstraint(accepted, ("start",))
    solutions = problem.getSolutions()
    if not solutions:
        return []

    ranked = sorted(
        solutions,
        key=lambda item: (
            abs((item["start"] - desired).total_seconds()),
            item["start"].date() != desired.date(),
        ),
    )
    results = []
    for item in ranked[:top_k]:
        start = item["start"]
        end = start + timedelta(minutes=duration_minutes)
        results.append(
            {
                "start_time": start.isoformat(timespec="seconds"),
                "end_time": end.isoformat(timespec="seconds"),
                "location": room_at(start, end),
                "distance_minutes": int(abs((start - desired).total_seconds()) / 60),
            }
        )
    return results


def format_alternative(alt: dict) -> str:
    """把求解器结果格式化成给用户看的紧凑文案。"""
    start = parse_datetime(alt["start_time"], "start_time")
    end = parse_datetime(alt["end_time"], "end_time")
    when = f"{start:%m.%d} {start:%H:%M}-{end:%H:%M}"
    room = alt.get("location") or "会议室待定"
    return f"{when}（{room}，无冲突）"


def format_recommendation(primary: dict | None, alternatives: list[dict]) -> str:
    """HITL 方案文案：具体推荐 + 无冲突说明 + 备选，不留空话。"""
    if not primary:
        return ""
    text = f"推荐 {format_alternative(primary)}，当前时段无会议室冲突、无参会人日程冲突。"
    others = [format_alternative(alt) for alt in alternatives[1:3]]
    if others:
        text += " 若该时段不合适，备选：" + "；".join(others) + "。"
    return text
