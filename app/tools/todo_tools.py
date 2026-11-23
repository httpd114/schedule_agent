"""待办类工具：创建待办、查询待办。入参 Schema 见 app/tools/schemas.py。"""
from __future__ import annotations

from app.models import repository
from app.tools.base import BaseTool, success_payload
from app.tools.registry import register_tool
from app.tools.schemas import CreateTodoArgs, QueryTodosArgs
from app.utils.validators import parse_optional_datetime


@register_tool
class CreateTodoTool(BaseTool):
    name = "create_todo"
    description = (
        "创建一条待办事项。用户说'记个待办''提醒我去做 X'时调用。"
        "deadline 为待办截止时间 ISO 8601（由时间解析层生成），非必填；优先级 high/medium/low。"
    )
    args_model = CreateTodoArgs

    def execute(
        self, *, content, deadline=None, priority=None, description=None,
        tags=None, link_schedule_id=None, **_
    ) -> str:
        due = parse_optional_datetime(deadline, "deadline", default_hour=18)
        if isinstance(tags, list):
            tags = ", ".join(str(t) for t in tags) or None
        if link_schedule_id:
            link = f"关联日程 #{link_schedule_id}"
            description = f"{description or ''}\n{link}".strip()
        record = repository.create_todo(
            title=content,
            due_time=due,
            priority=priority,
            description=description,
            tags=tags,
        )
        return success_payload(
            message=f"已创建待办 #{record['id']}：{record['title']}",
            todo=record,
        )


@register_tool
class QueryTodosTool(BaseTool):
    name = "query_todos"
    description = (
        "查询 context_window 时间区间内到期的待办列表。"
        "窗口由意图解析层按业务时间自动生成，不要接收用户原始时间文本。"
    )
    args_model = QueryTodosArgs

    def execute(
        self, *, context_window=None, keyword=None, status=None, due_before=None, limit=20, **_
    ) -> str:
        state = None if status in (None, "", "all") else status
        due = parse_optional_datetime(due_before, "due_before", default_hour=23)
        due_after = None
        if context_window:
            window = context_window if isinstance(context_window, dict) else {}
            if window.get("start"):
                due_after = parse_optional_datetime(window["start"], "context_window.start", default_hour=0)
            if window.get("end"):
                due = parse_optional_datetime(window["end"], "context_window.end", default_hour=23)
        records = repository.list_todos(
            keyword=keyword, status=state, due_before=due, due_after=due_after, limit=limit
        )
        return success_payload(
            message=f"共查询到 {len(records)} 条待办。",
            count=len(records),
            todos=records,
        )
