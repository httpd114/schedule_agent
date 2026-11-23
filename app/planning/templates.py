"""Planner 候选方案模板库。

优先输出对团队扰动最小的折中方案；时间编排类直接调用约束求解器，
给出【Top3 具体时段 + 可用会议室】，而不是“下午/下周”这种模糊建议。
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

from app.planning.constraint_solver import (
    DEFAULT_ROOMS,
    format_alternative,
    format_recommendation,
    suggest_windows,
)
from app.planning.rules import BusinessRule
from app.utils.validators import parse_datetime


def _preferred_location(intent: dict) -> str | None:
    """用户明确说出的会议室优先保留，只有冲突时才由求解器换房。"""
    text = f"{intent.get('raw', '')} {intent.get('summary', '')}"
    match = re.search(r"\d+[Ff]-[A-Za-z0-9]+", text)
    return match.group(0) if match else None


def _room_pool(intent: dict) -> list[str]:
    preferred = _preferred_location(intent)
    return ([preferred] if preferred else []) + DEFAULT_ROOMS


def _find_schedule(text: str, bundle: dict, keywords: tuple[str, ...]) -> dict | None:
    schedules = bundle.get("schedules", [])
    for word in keywords:
        for item in schedules:
            if word in str(item.get("title") or ""):
                return item
    return None


def _desired_start(intent: dict, fallback: datetime, days: int = 1) -> datetime:
    """优先读结构化 target_time；time_hint 仅为旧调用兼容。"""
    target = intent.get("target_time") or {}
    hint = (
        target.get("schedule_start")
        or target.get("leave_date")
        or target.get("check_time")
        or intent.get("time_hint")
    )
    if hint:
        try:
            return parse_datetime(hint, "target_time")
        except Exception:  # noqa: BLE001
            pass
    return fallback + timedelta(days=days)


def _window_source(intent: dict, bundle: dict, fallback_time: datetime) -> tuple[str, int]:
    desired = _desired_start(intent, fallback_time)
    start_raw = desired.isoformat(timespec="seconds")
    # 用户明确说了时长则优先；否则沿用原日程长度；都没有则 60 分钟
    duration = 60
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:小时|个小时|h\b)", f"{intent.get('raw', '')} {intent.get('summary', '')}")
    if match:
        duration = max(15, min(240, int(float(match.group(1)) * 60)))
    for item in bundle.get("schedules", []):
        if item.get("title") and str(item["title"]) in str(intent.get("summary", "")):
            try:
                start = parse_datetime(item["start_time"], "start_time")
                end = parse_datetime(item["end_time"], "end_time") if item.get("end_time") else start + timedelta(hours=1)
                duration = max(30, min(180, int((end - start).total_seconds() // 60)))
            except Exception:  # noqa: BLE001
                pass
            break
    return start_raw, duration


def _meeting_absence_plans(intent: dict, bundle: dict, rules: list[BusinessRule]) -> list[dict] | None:
    text = f"{intent.get('summary', '')} {intent.get('raw', '')}"
    absence = (
        ("无法参加" in text and "周会" in text)
        or ("参加不了" in text and "周会" in text)
        or ("不能参加" in text and "周会" in text)
        or ("缺席" in text and "周会" in text)
        or "周会冲突" in text
        or "周会改期" in text
    )
    if not absence:
        return None

    meeting = _find_schedule(text, bundle, ("周会",)) or {}
    meeting_id = meeting.get("id")
    meeting_start = None
    if meeting.get("start_time"):
        try:
            meeting_start = parse_datetime(meeting["start_time"], "start_time")
        except Exception:  # noqa: BLE001
            pass
    base = meeting_start or datetime.now() + timedelta(days=1)
    meeting_end = meeting_start + timedelta(hours=1) if meeting_start else base + timedelta(hours=1)
    minutes_due = (meeting_end + timedelta(hours=24)).isoformat(timespec="seconds")

    # 改期备选：求解器直接给具体时段 + 可用会议室
    start_raw, duration = _window_source(intent, bundle, base)
    alternatives = suggest_windows(start_raw, duration, bundle, rules, top_k=3, span_days=3, rooms=_room_pool(intent))
    top = alternatives[0] if alternatives else None
    recommendation = format_recommendation(top, alternatives)

    plan_a = {
        "id": "A",
        "title": "找同事代参会 + 会后同步纪要（扰动最小）",
        "description": "请一位同事代为参加周会，会后 24 小时内把结论、责任人和截止时间同步给你。",
        "steps": [
            {"tool": "create_todo", "args": {"content": "请同事代为参加周会并记录要点", "priority": "high"}},
            {"tool": "create_todo", "args": {"content": "会后 24 小时内同步纪要", "deadline": minutes_due, "priority": "high"}},
        ],
        "risks": ["需提前与代参会同事确认", "纪要时效按会议管理规范 24 小时执行"],
        "cost": "low",
        "start_time": base.isoformat(timespec="seconds"),
        "end_time": meeting_end.isoformat(timespec="seconds"),
        "recommendation": "无需改动会议室与他人日程；只需确认代参会人选，纪要最晚 " + minutes_due[:16].replace("T", " "),
    }
    plan_b = {
        "id": "B",
        "title": "提交请假 + 安排工作交接",
        "description": "按考勤流程提交请假（年假需提前 3 个工作日），并在此之前完成交接。",
        "steps": [
            {"tool": "create_todo", "args": {"content": "提交请假申请并在系统中跟踪销假", "priority": "high", "deadline": (base - timedelta(days=3)).isoformat(timespec="seconds")}},
            {"tool": "create_todo", "args": {"content": "完成周会相关事项交接", "priority": "medium"}},
        ],
        "risks": ["年假需提前 3 个工作日", "超过 3 天需部门负责人审批并 HR 备案"],
        "cost": "medium",
        "start_time": base.isoformat(timespec="seconds"),
        "end_time": meeting_end.isoformat(timespec="seconds"),
        "recommendation": "不改变会议安排，但需走审批链路并落实交接。",
    }
    steps_c = [{"tool": "create_todo", "args": {"content": "通知参与人周会改期并同步新会议室", "priority": "high"}}]
    if top:
        steps_c.insert(
            0,
            {
                "tool": "create_schedule",
                "args": {
                    "title": meeting.get("title") or "项目周会（改期）",
                    "start_time": top["start_time"],
                    "end_time": top["end_time"],
                    "room_prefer": top.get("location"),
                },
            },
        )
        if meeting_id:
            steps_c.append({"tool": "delete_schedule", "args": {"id": meeting_id}})
    plan_c = {
        "id": "C",
        "title": "会议整体改期（新建 + 自动取消原日程）",
        "description": "用约束求解器选出的无冲突时段重开会议，确认后原子地软删除原日程，便于回溯。",
        "steps": steps_c,
        "risks": ["需要通知全部参与人", "跨部门会议需至少提前 1 个工作日通知"],
        "cost": "high",
        "start_time": top["start_time"] if top else start_raw,
        "end_time": top["end_time"] if top else None,
        "reschedule_of": meeting_id if meeting_id and top else None,
        "recommendation": recommendation or "暂未找到完全无冲突的改期时段，需人工协调。",
    }
    return [plan_a, plan_b, plan_c]


def _leave_plans(intent: dict, bundle: dict, rules: list[BusinessRule]) -> list[dict] | None:
    summary = str(intent.get("summary", "")) + str(intent.get("raw", ""))
    if "请假" not in summary and "休假" not in summary:
        return None
    if _meeting_absence_plans(intent, bundle, rules):
        return None
    hint = (intent.get("target_time") or {}).get("leave_date") or intent.get("time_hint")
    base = _desired_start(intent, datetime.now(), days=3)
    start = base.isoformat(timespec="seconds")
    return [
        {
            "id": "A",
            "title": "事假 + 交接（无需提前 3 个工作日）",
            "description": "提交事假申请，先完成关键事项交接，由直属主管审批。",
            "steps": [
                {"tool": "create_todo", "args": {"content": "提交事假申请（含交接清单）", "priority": "high", "deadline": start}},
                {"tool": "create_todo", "args": {"content": "销假并更新进度看板", "priority": "medium"}},
            ],
            "risks": ["1 至 3 天需部门负责人审批"],
            "cost": "low",
            "recommendation": f"对团队扰动最小：不改变既有日程，请假申请最晚 {start[:16].replace('T', ' ')} 前提交。" + (f"（用户指定：{hint}）" if hint else ""),
        },
        {
            "id": "B",
            "title": "年假 + 提前 3 个工作日申请",
            "description": "按考勤制度提前 3 个工作日提交年假，期间安排备份人。",
            "steps": [
                {"tool": "create_todo", "args": {"content": "提前 3 个工作日提交年假申请", "priority": "high", "deadline": start}},
                {"tool": "create_todo", "args": {"content": "指定年假期间备份人并完成交接", "priority": "medium"}},
            ],
            "risks": ["年假需提前 3 个工作日", "超过 3 天须 HR 备案"],
            "cost": "medium",
            "recommendation": "合规度最高，但需要提前申请并指定备份人。",
        },
        {
            "id": "C",
            "title": "调休 + 事后补登记",
            "description": "用调休抵扣，事后在系统登记并同步进度看板。",
            "steps": [
                {"tool": "create_todo", "args": {"content": "提交调休申请并登记销假", "priority": "medium", "deadline": start}},
            ],
            "risks": ["节假日调休以公司公告为准"],
            "cost": "high",
            "recommendation": "适合已有加班余额的场景，需以公司公告为准。",
        },
    ]


def _reschedule_plans(intent: dict, bundle: dict, rules: list[BusinessRule]) -> list[dict] | None:
    summary = str(intent.get("summary", "")) + str(intent.get("raw", ""))
    if not any(word in summary for word in ("改期", "调整到", "推迟", "提前到", "换到")):
        return None
    if _meeting_absence_plans(intent, bundle, rules):
        return None
    original = None
    for item in bundle.get("schedules", []):
        title = str(item.get("title") or "")
        if title and any(ch in title for ch in summary):
            original = item
            break
    if not original:
        original = bundle.get("schedules", [{}])[0] if bundle.get("schedules") else None
    if not original or not original.get("id"):
        return None
    try:
        original_start = parse_datetime(original["start_time"], "start_time")
        original_end = parse_datetime(original["end_time"], "end_time") if original.get("end_time") else original_start + timedelta(hours=1)
        duration = max(30, int((original_end - original_start).total_seconds() // 60))
    except Exception:  # noqa: BLE001
        original_start = datetime.now() + timedelta(days=1)
        duration = 60
    start_raw = _desired_start(intent, original_start).isoformat(timespec="seconds")
    alternatives = suggest_windows(start_raw, duration, bundle, rules, top_k=3, span_days=3, rooms=_room_pool(intent))
    if not alternatives:
        return None
    plans = []
    for index, alt in enumerate(alternatives, 1):
        rotated = alternatives[index - 1:] + alternatives[:index - 1]
        plans.append(
            {
                "id": chr(64 + index),
                "title": f"{original['title']} · 改期备选 {index}",
                "description": "确认后新建日程并在同一事务中软删除原日程，可回溯。",
                "steps": [
                    {
                        "tool": "create_schedule",
                        "args": {
                            "title": original["title"],
                            "start_time": alt["start_time"],
                            "end_time": alt["end_time"],
                            "room_prefer": alt.get("location"),
                        },
                    },
                    {"tool": "delete_schedule", "args": {"id": original["id"]}},
                ],
                "risks": ["需通知参与人新时间", "原日程会变为已取消状态，可回溯"],
                "cost": "low" if index == 1 else ("medium" if index == 2 else "high"),
                "start_time": alt["start_time"],
                "end_time": alt["end_time"],
                "reschedule_of": original["id"],
                "recommendation": format_recommendation(alt, rotated),
            }
        )
    return plans


def _time_window_plans(intent: dict, bundle: dict, rules: list[BusinessRule]) -> list[dict] | None:
    target = intent.get("target_time") or {}
    has_time = target.get("schedule_start") or intent.get("time_hint")
    if intent.get("category") not in {"create", "plan"} or not has_time:
        return None
    start_raw, duration = _window_source(intent, bundle, datetime.now())
    alternatives = suggest_windows(start_raw, duration, bundle, rules, top_k=3, span_days=5, rooms=_room_pool(intent))
    if not alternatives:
        return None
    title = str(intent.get("summary") or "新日程")[:40]
    plans = []
    for index, alt in enumerate(alternatives, 1):
        rotated = alternatives[index - 1:] + alternatives[:index - 1]
        steps = [
            {
                "tool": "create_schedule",
                "args": {
                    "title": title,
                    "start_time": alt["start_time"],
                    "end_time": alt["end_time"],
                    "room_prefer": alt.get("location"),
                },
            }
        ]
        if index == 1:
            # 纪要待办必须写入自身 deadline（会议 end_time + 24h），公式与会议管理规范一致。
            # 一旦写入，该 deadline 就是待办唯一可信时间源，与关联日程再无筛选关系。
            minutes_due = (
                parse_datetime(alt["end_time"], "end_time") + timedelta(hours=24)
            ).isoformat(timespec="seconds")
            steps.append({
                "tool": "create_todo",
                "args": {
                    "content": "会后 24 小时内补充纪要要点",
                    "deadline": minutes_due,
                    "priority": "medium",
                },
            })
        plans.append(
            {
                "id": chr(64 + index),
                "title": f"{title} · {'推荐时段' if index == 1 else f'备选 {index}'}",
                "description": "时间窗口与会议室由约束求解器校验，避开既有日程与参会人冲突。",
                "steps": steps,
                "risks": ["如参与人有临时变动需再确认"] if index == 1 else [f"与首选时段相差 {alt['distance_minutes']} 分钟"],
                "cost": "low" if index == 1 else ("medium" if index == 2 else "high"),
                "start_time": alt["start_time"],
                "end_time": alt["end_time"],
                "recommendation": format_recommendation(alt, rotated),
            }
        )
    return plans


def build_templates(intent: dict, bundle: dict, rules: list[BusinessRule]) -> list[dict] | None:
    """命中模板则返回确定性方案（已含 Top3 时段/会议室），否则返回 None 交给 LLM。"""
    for builder in (_meeting_absence_plans, _leave_plans, _reschedule_plans, _time_window_plans):
        plans = builder(intent, bundle, rules)
        if plans:
            return plans
    return None


def template_hint(intent: dict, bundle: dict, rules: list[BusinessRule]) -> str:
    """给 LLM 的提示：哪些轻量方案优先、有哪些具体可用时段。"""
    start_raw, duration = _window_source(intent, bundle, datetime.now())
    alternatives = suggest_windows(start_raw, duration, bundle, rules, top_k=3, span_days=3, rooms=_room_pool(intent))
    if not alternatives:
        return "（约束求解器未找到完全无冲突时段，请给出需要人工协调的说明）"
    return "约束求解器给出的无冲突时段（可直接用于方案，务必带具体会议室）：" + "；".join(
        format_alternative(alt) for alt in alternatives
    )
