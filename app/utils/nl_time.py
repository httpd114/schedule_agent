"""自然语言时间解析层。

职责边界（本次重构的核心）：
- ``context_window`` 是**聚合查询窗口**，只能由本模块从业务时间自动推导，
  绝不接收用户原始自然语言时间串。
- ``target_time`` 分组存放业务时间（会议起止、待办截止、请假日期、查会议室单点），
  由对应工具的 Pydantic Schema 接收 ISO 8601 值。

对外入口：``parse_nl_time(text, intent_type)``，其中 text 可为单个表达或
{语义键: 原文} 字典。返回结构化 datetime，调用方再转成 ISO 写入 State。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from app.utils.validators import ValidationError, _parse_clock, parse_datetime

# target_time 的合法键（leave_end 为请假区间的结束，属于扩展字段）
TARGET_KEYS = (
    "schedule_start", "schedule_end",
    "todo_deadline", "leave_date", "leave_end", "check_time",
)

_WEEKDAY_CN = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}

# 单个时间原文按意图类型落到哪个语义键
_SINGLE_KEY = {
    "create_schedule": "schedule_start",
    "reschedule": "schedule_start",
    "create_todo": "todo_deadline",
    "apply_leave": "leave_date",
    "check_room": "check_time",
    "query_schedule": "context_start",
    "query_todo": "context_start",
    "query_leave": "context_start",
    "policy": "context_start",
}

# 右侧只允许“时钟”或“周X”，避免把 2026-11-22 中的连字符当作区间分隔符
_RANGE_RE = re.compile(
    r"^(?P<left>.+?)\s*(?:到|至|[-–—~～])\s*"
    r"(?P<right>(?:上午|下午|晚上|中午|凌晨)?\s*\d{1,2}(?::\d{2})?(?:点|时|半)?|周[一二三四五六日天])$"
)
_WEEK_RE = re.compile(r"^(下|本|这)周([一二三四五六日天])(?:\s*(.*))?$")
_SHORT_WEEK_RE = re.compile(r"^周([一二三四五六日天])(?:\s*(.*))?$")
_MONTH_DAY_RE = re.compile(
    r"^(?:(\d{4})[年/])?(\d{1,2})[月./-](\d{1,2})(?:日|号)?(?:\s*(.*))?$"
)
_PREFIX_RE = re.compile(r"^(?:截止到?|最晚|不晚于|之前|在|于|到)\s*")


@dataclass
class TimeParseResult:
    """一次解析的完整产物：查询窗口 + 业务时间，均为 datetime。"""

    intent_type: str
    context_window: dict[str, datetime | None] = field(
        default_factory=lambda: {"start": None, "end": None}
    )
    target_time: dict[str, datetime | None] = field(
        default_factory=lambda: {key: None for key in TARGET_KEYS}
    )
    raw_hits: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_state(self) -> tuple[dict[str, str | None], dict[str, str | None]]:
        """转成可安全放进 LangGraph State 的 ISO 字符串（绝不外泄原文）。"""
        return (
            {key: _iso(value) for key, value in self.context_window.items()},
            {key: _iso(value) for key, value in self.target_time.items()},
        )


def _iso(value: datetime | None) -> str | None:
    return value.isoformat(timespec="seconds") if value else None


def _split_range(text: str) -> tuple[str, str] | None:
    """拆 '下周四下午14:00–15:30' / '11月26日到28日' / '14:00-15:30'。"""
    match = _RANGE_RE.match(text.strip())
    if not match:
        return None
    return match.group("left").strip(), match.group("right").strip()


def _combine(day, rest: str, default_hour: int, now: datetime) -> datetime:
    """日期 + 可选时钟表达。"""
    from datetime import time as _time

    if rest and rest.strip():
        hour, minute = _parse_clock(rest)
        return datetime.combine(day, _time(hour, minute))
    return datetime.combine(day, _time(default_hour, 0))


def _parse_point(
    text: str,
    base_date,
    now: datetime,
    default_hour: int = 9,
    week_anchor: datetime | None = None,
) -> datetime | None:
    """把一个时间点表达解析成 datetime；无法识别时返回 None，不抛异常。"""
    cleaned = _PREFIX_RE.sub("", str(text or "").strip())
    if not cleaned:
        return None

    # 1) ISO 8601 / 完整中文日期 —— 优先，避免日期里的 '-' 被误当区间
    try:
        return parse_datetime(cleaned, "time", default_hour=default_hour, now=now)
    except ValidationError:
        pass

    # 2) 下周四 / 本周一 / 这周五 14:00
    match = _WEEK_RE.match(cleaned)
    if match:
        prefix, weekday, rest = match.groups()
        monday = (week_anchor.date() if week_anchor else now.date()) - timedelta(
            days=(week_anchor.date() if week_anchor else now.date()).weekday()
        )
        if prefix != "下" and week_anchor is None:
            # “周一 10:00”若今天已过该星期，按最近的下一个同星期理解
            candidate = monday + timedelta(days=_WEEKDAY_CN[weekday])
            if candidate < now.date():
                monday += timedelta(days=7)
        day = monday + timedelta(days=7 if prefix == "下" else 0) + timedelta(days=_WEEKDAY_CN[weekday])
        return _combine(day, rest or "", default_hour, now)

    # 3) 12月2号 / 11.26 14:00 / 2026/11/26 14:00
    match = _MONTH_DAY_RE.match(cleaned)
    if match:
        year_s, month_s, day_s, rest = match.groups()
        year = int(year_s) if year_s else now.year
        try:
            day = datetime(year, int(month_s), int(day_s)).date()
        except ValueError:
            return None
        return _combine(day, rest or "", default_hour, now)

    # 4) 周四 14:00（无“下周”修饰）
    match = _SHORT_WEEK_RE.match(cleaned)
    if match:
        weekday, rest = match.groups()
        monday = (week_anchor.date() if week_anchor else now.date()) - timedelta(
            days=(week_anchor.date() if week_anchor else now.date()).weekday()
        )
        day = monday + timedelta(days=_WEEKDAY_CN[weekday])
        if week_anchor is None and day < now.date():
            day += timedelta(days=7)
        return _combine(day, rest or "", default_hour, now)

    # 5) 纯时钟：14:00 / 下午3点 / 晚上8点半，基准日期由调用方给出
    try:
        hour, minute = _parse_clock(cleaned)
        from datetime import time as _time

        return datetime.combine(base_date, _time(hour, minute))
    except ValidationError:
        return None


def _resolve_points(
    raw: str,
    base_date,
    now: datetime,
    default_hour: int = 9,
    as_day_range: bool = False,
) -> tuple[datetime | None, datetime | None]:
    """返回 (起点, 终点)；单点表达终点为 None（查询类补成当天 23:59:59）。"""
    text = str(raw or "").strip()
    parts = _split_range(text)
    if parts:
        start = _parse_point(parts[0], base_date, now, default_hour)
        end = _parse_point(
            parts[1], start.date() if start else base_date, now,
            default_hour=18, week_anchor=start,
        )
    else:
        start = _parse_point(text, base_date, now, default_hour)
        end = None

    if as_day_range and start is not None and end is None:
        end = start.replace(hour=23, minute=59, second=59)
    return start, end


def _infer_window(
    target: dict[str, datetime | None], now: datetime
) -> tuple[datetime | None, datetime | None]:
    """按业务时间自动推导查询窗口：会议/待办/请假 ±1 天，查会议室仅当天。"""
    check = target.get("check_time")
    if check:
        return (
            check.replace(hour=0, minute=0, second=0, microsecond=0),
            check.replace(hour=23, minute=59, second=59, microsecond=0),
        )
    center = (
        target.get("schedule_start")
        or target.get("todo_deadline")
        or target.get("leave_date")
    )
    if not center:
        return None, None
    last = target.get("schedule_end") or target.get("leave_end") or center
    return (
        (center - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0),
        (last + timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=0),
    )


def parse_nl_time(
    text: str | dict[str, Any],
    intent_type: str,
    *,
    now: datetime | None = None,
    base_date=None,
) -> TimeParseResult:
    """把用户时间原文解析为结构化 datetime，按意图分流入不同字段。

    - 查询类（query_*）：结果进 context_window，target_time 全空。
    - 创建/变更类：业务时间进 target_time，context_window 由程序自动推导。
    - admin / policy：完全不解析时间，返回空结果。
    """
    now = now or datetime.now()
    base = base_date or now.date()
    result = TimeParseResult(intent_type=intent_type)

    if intent_type in {"admin", "policy"}:
        return result

    if isinstance(text, dict):
        expressions = {str(key): value for key, value in text.items() if value}
    else:
        key = _SINGLE_KEY.get(intent_type, "context_start")
        expressions = {key: text} if text else {}

    for key, raw in expressions.items():
        result.raw_hits[key] = str(raw)
        if key.startswith("context_"):
            # 查询窗口必须是完整自然日：起点 00:00、终点 23:59:59。
            # 若用默认 9 点作起点，早上 8 点截止的待办会被错误排除。
            start, end = _resolve_points(
                str(raw), base, now, default_hour=0, as_day_range=True
            )
            if start:
                result.context_window["start"] = start
                result.context_window["end"] = end
            continue

        if key not in TARGET_KEYS:
            continue
        hour = 18 if key in {"todo_deadline", "schedule_end"} else 9
        start, end = _resolve_points(str(raw), base, now, default_hour=hour)
        if start:
            result.target_time[key] = start
        if key == "schedule_start" and end and not result.target_time["schedule_end"]:
            result.target_time["schedule_end"] = end
        if key == "leave_date" and end and not result.target_time["leave_end"]:
            result.target_time["leave_end"] = end

    if not result.context_window["start"]:
        start, end = _infer_window(result.target_time, now)
        result.context_window["start"] = start
        result.context_window["end"] = end
    return result
