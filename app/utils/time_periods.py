"""模糊时段偏好：中文时段词 -> 固定区间映射（单一事实来源）。

parse_nl_time 解析出 prefer 结构后，Planner / 工具层用这里的区间把
「日期 + 上午/下午/晚上」转成 CSP 检索边界，不再强行补一个精确的
start/end，也就不会对“下午”这类表达抛 ValidationError。
"""
from __future__ import annotations

import re
from datetime import date, datetime, time as _time, timedelta
from typing import Any

# 用户约定：morning 08:00-12:00 / afternoon 13:00-18:00 / evening 18:00-21:00
PERIOD_BOUNDS: dict[str, tuple[int, int]] = {
    "morning": (8, 12),
    "afternoon": (13, 18),
    "evening": (18, 21),
}

PERIOD_WORDS: dict[str, str] = {
    "早上": "morning", "早晨": "morning", "上午": "morning",
    "中午": "afternoon", "下午": "afternoon",
    "晚上": "evening", "傍晚": "evening", "晚间": "evening", "夜里": "evening",
}

# 具体时钟标志：出现这些就视为 exact，不走时段偏好
_CLOCK_HINT_RE = re.compile(r"\d{1,2}\s*(?::\s*\d{2}|点|时|半)")
_ROOM_RE = re.compile(r"\d+[Ff]-[A-Za-z0-9]+")


def period_of(text: str) -> str | None:
    """返回文本里出现的时段枚举；无时段词返回 None。"""
    value = str(text or "")
    for word, period in PERIOD_WORDS.items():
        if word in value:
            return period
    return None


def is_period_only(text: str) -> bool:
    """仅含时段词、不含具体时钟 —— 这类表达应产出 prefer 而非 exact。"""
    value = str(text or "")
    return period_of(value) is not None and _CLOCK_HINT_RE.search(value) is None


def strip_period(text: str) -> str:
    value = str(text or "")
    for word in PERIOD_WORDS:
        value = value.replace(word, "")
    return value.strip()


def extract_room(text: str) -> str | None:
    match = _ROOM_RE.search(str(text or ""))
    return match.group(0) if match else None


def period_window(day: date | datetime, period: str) -> tuple[datetime, datetime]:
    """把某天的时段枚举展开成 [start, end) 区间。"""
    if isinstance(day, datetime):
        day = day.date()
    lower, upper = PERIOD_BOUNDS.get(period, PERIOD_BOUNDS["afternoon"])
    return (
        datetime.combine(day, _time(lower, 0)),
        datetime.combine(day, _time(upper, 0)),
    )


def as_prefer(
    day: date | datetime,
    period: str,
    room_prefer: str | None = None,
) -> dict[str, Any]:
    """构造 prefer 输出结构：{type, date, period, room_prefer}。"""
    if isinstance(day, datetime):
        day = day.date()
    return {
        "type": "prefer",
        "date": day.isoformat(),
        "period": period,
        "room_prefer": room_prefer,
    }


def is_prefer(value: Any) -> bool:
    return isinstance(value, dict) and value.get("type") == "prefer"


def shift_day(day: date | datetime, days: int) -> date:
    if isinstance(day, datetime):
        day = day.date()
    return day + timedelta(days=days)
