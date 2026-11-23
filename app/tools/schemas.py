"""工具入参 Pydantic V2 Schema（按时间语义拆分）。

注册表直接从这些模型生成 Function Calling 描述；校验失败时输出结构化
validation_error（含 errors 列表与修复建议），供 LLM 自愈修正。

时间字段分工：
- query_schedules / query_todos -> context_window（程序生成的查询区间，TimeRange）
- create_schedule              -> start_time / end_time（会议业务时间）
- create_todo                  -> deadline（待办截止）
- apply_leave                  -> leave_date（请假日期）
- check_room_occupancy         -> check_datetime（单点查询时间）

任何工具都不再有承接所有时间文本的 focus_time 字段。
"""
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import AliasChoices, BaseModel, BeforeValidator, ConfigDict, Field

from app.utils.validators import ValidationError as TimeValidationError
from app.utils.validators import parse_datetime


def _normalize_time(value: Any, intent_type: str) -> Any:
    """把合法时间表达归一成 ISO 字符串；非法输入转成 Pydantic 可识别的 ValueError。

    主路径是 ISO 8601；当 LLM 偶尔漏掉解析层、直接给出“11月26日”这类中文表达时，
    用 parse_nl_time 做一次兜底解析，避免不必要的校验失败与重试。
    """
    if value is None or value == "":
        return value
    try:
        return parse_datetime(value).isoformat(timespec="seconds")
    except TimeValidationError as exc:
        try:
            from datetime import datetime as _datetime

            from app.utils.nl_time import parse_nl_time

            parsed = parse_nl_time(str(value), intent_type, now=_datetime.now())
            fallback = (
                parsed.target_time.get("schedule_start")
                or parsed.target_time.get("todo_deadline")
                or parsed.context_window.get("start")
            )
            if fallback is not None:
                return fallback.isoformat(timespec="seconds")
        except Exception:  # noqa: BLE001 - 兜底解析失败仍走结构化错误
            pass
        raise ValueError(exc.hint or exc.message) from exc


def normalize_time(value: Any) -> Any:
    return _normalize_time(value, "create_schedule")


def normalize_deadline(value: Any) -> Any:
    """待办截止时间专属校验：中文日期默认补到当天 18:00。"""
    return _normalize_time(value, "create_todo")


TimeStr = Annotated[str, BeforeValidator(normalize_time)]
DeadlineStr = Annotated[str, BeforeValidator(normalize_deadline)]


def split_list(value: Any) -> Any:
    """LLM 偶尔把 tags/attendees 写成逗号字符串，这里宽容地归一成列表。"""
    if isinstance(value, str):
        return [part.strip() for part in value.replace("，", ",").split(",") if part.strip()]
    return value


StringList = Annotated[list[str] | None, BeforeValidator(split_list)]


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class TimeRange(ToolArgs):
    """聚合查询窗口，只能由程序从业务时间推导，不接受原始自然语言。"""

    start: TimeStr | None = Field(default=None, description="窗口起点 ISO 8601，程序生成")
    end: TimeStr | None = Field(default=None, description="窗口终点 ISO 8601，程序生成")


class CreateScheduleArgs(ToolArgs):
    title: str = Field(description="日程标题，例如 Q4 项目复盘会")
    start_time: TimeStr = Field(description="会议开始时间 ISO 8601，如 2026-11-22T14:30:00")
    end_time: TimeStr | None = Field(default=None, description="会议结束时间，格式同 start_time，可选")
    room_prefer: str | None = Field(
        default=None,
        validation_alias=AliasChoices("room_prefer", "location"),
        description="用户指定/求解器推荐的会议室，如 15F-B205，可选",
    )
    description: str | None = Field(default=None, description="日程备注，可选")
    attendees: StringList = Field(default=None, description="参与人姓名列表，可选")


class QuerySchedulesArgs(ToolArgs):
    context_window: TimeRange | None = Field(
        default=None, description="聚合查询窗口（start/end），优先于 start_from/start_to"
    )
    keyword: str | None = Field(default=None, description="标题关键词，可选")
    date: TimeStr | None = Field(default=None, description="查询某一天，可选")
    start_from: TimeStr | None = Field(default=None, description="兼容字段：区间起始，可选")
    start_to: TimeStr | None = Field(default=None, description="兼容字段：区间结束，可选")
    limit: int = Field(default=20, ge=1, le=50, description="最多返回条数")


class CreateTodoArgs(ToolArgs):
    content: str = Field(
        min_length=2, max_length=200,
        validation_alias=AliasChoices("content", "title"),
        description="待办内容，例如 会后24小时内补充纪要",
    )
    deadline: DeadlineStr | None = Field(
        default=None,
        validation_alias=AliasChoices("deadline", "due_time"),
        description="待办截止时间 ISO 8601，可选",
    )
    priority: Literal["high", "medium", "low"] = Field(default="medium", description="优先级")
    description: str | None = Field(default=None, description="详细说明，可选")
    tags: StringList = Field(default=None, description="标签列表，可选")
    link_schedule_id: int | None = Field(default=None, ge=1, description="关联的日程 ID，可选")


class QueryTodosArgs(ToolArgs):
    context_window: TimeRange | None = Field(
        default=None, description="聚合查询窗口（start/end），用于按截止时间过滤"
    )
    keyword: str | None = Field(default=None, description="标题关键词，可选")
    status: Literal["pending", "in_progress", "done", "all"] = Field(default="all", description="状态过滤")
    due_before: TimeStr | None = Field(default=None, description="兼容字段：该时间前到期，可选")
    limit: int = Field(default=20, ge=1, le=50, description="最多返回条数")


class ApplyLeaveArgs(ToolArgs):
    leave_date: TimeStr = Field(description="请假起始日期 ISO 8601，由解析层从用户表达生成")
    leave_type: Literal["事假", "年假", "病假", "调休"] = Field(default="事假", description="请假类型")
    reason: str | None = Field(default=None, description="请假事由，可选")


class CheckRoomOccupancyArgs(ToolArgs):
    check_datetime: TimeStr = Field(description="要查询的单点时间 ISO 8601")
    room_name: str = Field(min_length=2, description="会议室名称，如 15F-B205")


class SearchDocumentsArgs(ToolArgs):
    query: str = Field(min_length=1, max_length=500, description="检索问题或关键词")
    top_k: int = Field(default=3, ge=1, le=10, description="返回文档条数")


class DeleteScheduleArgs(ToolArgs):
    id: int = Field(gt=0, description="要取消的日程 ID；执行软删除，状态变为 cancelled，可回溯")
    reason: str | None = Field(default=None, description="取消原因，可选")


class RescheduleScheduleArgs(ToolArgs):
    """独立改期工具入参：软删除原日程 + 创建新日程，原子事务完成。

    target_date / period_prefer / room_prefer 承接 ug1 的 prefer 偏好原文语义，
    不做 Pydantic 时间校验；真正的写入时间由 CSP 求出后填入 start_time/end_time。
    """
    schedule_id: int = Field(
        gt=0,
        validation_alias=AliasChoices("schedule_id", "id"),
        description="要改期的原日程 ID",
    )
    target_date: str | None = Field(default=None, description="目标日期，如 2026-11-27")
    period_prefer: Literal["morning", "afternoon", "evening"] | None = Field(
        default=None, description="时段偏好：morning/afternoon/evening"
    )
    room_prefer: str | None = Field(default=None, description="偏好会议室，如 12F-A301")
    start_time: TimeStr | None = Field(default=None, description="求解器确定的新开始时间 ISO 8601")
    end_time: TimeStr | None = Field(default=None, description="求解器确定的新结束时间 ISO 8601")


class DeleteTodoArgs(ToolArgs):
    id: int = Field(gt=0, description="要删除的待办 ID；执行软删除（is_deleted=True），可后台找回")
    reason: str | None = Field(default=None, description="删除原因，可选")


def _field_names(model: type[ToolArgs]) -> dict[str, str]:
    """外部可传入的名字 -> 模型字段名，兼容 AliasChoices 中的旧名。"""
    mapping: dict[str, str] = {}
    for name, field in model.model_fields.items():
        mapping[name] = name
        alias = field.validation_alias
        if isinstance(alias, AliasChoices):
            mapping.update({str(choice): name for choice in alias.choices if isinstance(choice, str)})
        elif isinstance(alias, str):
            mapping[alias] = name
    return mapping


def sanitize_args(model: type[ToolArgs], arguments: dict) -> dict:
    """重试专用：剔除多余 key、归一化 ISO 时间、把数字字符串转成 int。

    严格模式仍由 BaseTool 的 extra='forbid' 负责；本函数只用于自动自愈重试，
    不会放宽对最终入参的 Pydantic 校验。旧字段名（title/due_time/location）
    会被映射到当前字段名，避免自愈时把可修复参数误删。
    """
    import json as _json

    allowed = _field_names(model)
    cleaned: dict[str, object] = {}
    for key, value in (arguments or {}).items():
        field_name = allowed.get(key)
        if field_name is None:
            continue
        field = model.model_fields.get(field_name)
        annotation = str(getattr(field, "annotation", ""))
        if "int" in annotation and isinstance(value, str) and value.strip().isdigit():
            value = int(value.strip())
        time_keys = ("time", "date", "deadline", "start", "end", "start_from", "start_to", "due_before")
        if isinstance(value, str) and any(token in field_name for token in time_keys):
            try:
                value = normalize_time(value)
            except ValueError:
                pass
        cleaned[field_name] = value
    # 深拷贝，避免调用方参数被就地修改
    return _json.loads(_json.dumps(cleaned, ensure_ascii=False, default=str))


def _clean_schema(node: Any) -> Any:
    """递归清理 JSON Schema，使其符合 Function Calling 规范。"""
    if isinstance(node, dict):
        if "anyOf" in node:
            variants = [item for item in node["anyOf"] if item.get("type") != "null"]
            if len(variants) == 1:
                merged = {**node, **variants[0]}
                merged.pop("anyOf", None)
                node = merged
        if node.get("default", "__missing__") is None:
            node.pop("default", None)
        for key, value in list(node.items()):
            node[key] = _clean_schema(value)
    elif isinstance(node, list):
        node = [_clean_schema(item) for item in node]
    return node


def _inline_refs(node: Any, defs: dict) -> Any:
    """内联展开 $defs/$ref：Function Calling 接口通常不接受 JSON Pointer 引用。"""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            target = defs.get(ref.rsplit("/", 1)[-1], {})
            merged = {**target, **{key: value for key, value in node.items() if key != "$ref"}}
            return _inline_refs(merged, defs)
        return {key: _inline_refs(value, defs) for key, value in node.items()}
    if isinstance(node, list):
        return [_inline_refs(item, defs) for item in node]
    return node


function_parameters_cache: dict[type, dict] = {}


def function_parameters(model: type[ToolArgs]) -> dict:
    """从 Pydantic 模型生成 Function Calling JSON Schema（内联、带缓存）。"""
    if model not in function_parameters_cache:
        raw = model.model_json_schema()
        defs = _clean_schema(raw.pop("$defs", {}))
        schema = _inline_refs(_clean_schema(raw), defs)
        schema.pop("title", None)  # 只移除模型根标题，不能误删名为 title 的字段
        schema.setdefault("additionalProperties", False)
        function_parameters_cache[model] = schema
    return function_parameters_cache[model]
