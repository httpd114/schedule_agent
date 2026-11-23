"""工具入参校验：时间格式归一化与枚举校验。

校验失败抛出 ValidationError，由工具基类统一转成结构化错误回传给模型，
让 LLM 有机会依据 hint 自行修正参数后重试。
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta


class ValidationError(ValueError):
    """参数校验失败，携带可供模型自愈的提示。"""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.message = message
        self.hint = hint


_WEEKDAY_CN = {"今天": 0, "今日": 0, "明天": 1, "明日": 1, "后天": 2, "大后天": 3}
_CLOCK_RE = re.compile(r"^(上午|下午|晚上|中午|凌晨)?\s*(\d{1,2})(?::(\d{1,2}))?\s*(?:点|时)?\s*(半)?$")
_DATE_RE = re.compile(r"^(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})日?$")
_DATE_TIME_RE = re.compile(r"^(\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?)\s+(.+)$")


def _parse_clock(text: str) -> tuple[int, int]:
    """解析 '下午3点' / '9:30' / '晚上8点半' 等中文时钟表达。"""
    m = _CLOCK_RE.match(text.strip())
    if not m:
        raise ValidationError(
            f"无法解析时间 '{text}'",
            "请使用 HH:MM（如 14:30）或 ISO 8601（如 2026-11-22T14:30:00）。",
        )
    period, hour_s, minute_s, half = m.groups()
    hour = int(hour_s)
    minute = int(minute_s) if minute_s else (30 if half else 0)
    if period in ("下午", "晚上") and hour < 12:
        hour += 12
    elif period == "中午" and hour < 11:
        hour += 12
    elif period == "凌晨" and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValidationError(f"时间超出有效范围: '{text}'", "小时需在 0-23，分钟需在 0-59。")
    return hour, minute


def _parse_date(text: str) -> datetime | None:
    m = _DATE_RE.match(text.strip())
    if not m:
        return None
    year, month, day = (int(part) for part in m.groups())
    try:
        return datetime(year, month, day)
    except ValueError:
        return None


def parse_datetime(
    value: object,
    field_name: str = "time",
    *,
    default_hour: int = 9,
    now: datetime | None = None,
) -> datetime:
    """把模型生成的时间字符串解析成 datetime。

    支持 ISO 8601、中文日期、以及“今天/明天 14:00”“明天下午3点”等相对表达；
    纯日期按 default_hour 补齐时间。
    """
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(
            f"字段 {field_name} 缺失或不是有效字符串: {value!r}",
            f"请为 {field_name} 提供如 '2026-11-22T14:30:00' 的 ISO 8601 时间。",
        )

    text = value.strip()
    today = (now or datetime.now()).replace(hour=0, minute=0, second=0, microsecond=0)

    # 1) 相对日期前缀，例如 明天 14:00 / 后天下午3点
    for word, offset in _WEEKDAY_CN.items():
        if text.startswith(word):
            rest = text[len(word) :].strip()
            if not rest:
                return today + timedelta(days=offset, hours=default_hour)
            hour, minute = _parse_clock(rest)
            return today + timedelta(days=offset, hours=hour, minutes=minute)

    # 2) 纯日期
    date_only = _parse_date(text)
    if date_only is not None:
        return date_only.replace(hour=default_hour)

    # 3) 日期 + 时钟，例如 2026-11-22 14:30 / 2026年11月22日 下午3点
    m = _DATE_TIME_RE.match(text)
    if m:
        base = _parse_date(m.group(1))
        if base is not None:
            hour, minute = _parse_clock(m.group(2))
            return base.replace(hour=hour, minute=minute)

    # 4) ISO 8601（fromisoformat 兼容 'T' 与 ' ' 分隔）
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        pass

    raise ValidationError(
        f"字段 {field_name} 的时间格式无法解析: '{value}'",
        f"请使用 ISO 8601 格式，例如 '{(today + timedelta(days=1)).date()}T14:30:00'，"
        "也支持 '明天 14:30'。",
    )


def parse_optional_datetime(value: object, field_name: str, **kwargs) -> datetime | None:
    if value in (None, "", "null"):
        return None
    return parse_datetime(value, field_name, **kwargs)


def normalize_choice(value: object, allowed: set[str], field_name: str, default: str) -> str:
    if value in (None, ""):
        return default
    text = str(value).strip()
    if text in allowed:
        return text
    raise ValidationError(
        f"字段 {field_name} 取值 '{value}' 不在允许范围内",
        f"请从 {sorted(allowed)} 中选择。",
    )


def require_string(value: object, field_name: str, max_length: int = 200) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(
            f"字段 {field_name} 不能为空",
            f"请提供有效的 {field_name} 文本。",
        )
    text = value.strip()
    if len(text) > max_length:
        raise ValidationError(
            f"字段 {field_name} 过长（{len(text)} > {max_length}）",
            f"请将 {field_name} 精简到 {max_length} 字以内。",
        )
    return text
