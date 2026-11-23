"""请假类工具：把解析后的请假日期登记为审批待办。入参 Schema 见 app/tools/schemas.py。"""
from __future__ import annotations

from app.models import repository
from app.tools.base import BaseTool, success_payload
from app.tools.registry import register_tool
from app.tools.schemas import ApplyLeaveArgs
from app.utils.validators import parse_datetime


@register_tool
class ApplyLeaveTool(BaseTool):
    name = "apply_leave"
    description = (
        "登记一条请假申请待办。用于用户说'下周一到周三请假'时。"
        "leave_date 是解析层生成的请假起始日期 ISO 8601；leave_type 取 事假/年假/病假/调休。"
        "本工具只记录申请任务，不绕过制度审批，也不写入日程表。"
    )
    args_model = ApplyLeaveArgs

    def execute(self, *, leave_date, leave_type="事假", reason=None, **_) -> str:
        start = parse_datetime(leave_date, "leave_date")
        title = f"提交{leave_type}申请（{start.date().isoformat()}）"
        description = f"请假起始：{start.isoformat(timespec='seconds')}"
        if reason:
            description += f"\n事由：{reason}"
        record = repository.create_todo(
            title=title,
            due_time=start,
            priority="high",
            description=description,
            tags=f"请假,{leave_type}",
        )
        return success_payload(
            message=f"已登记{leave_type}申请待办 #{record['id']}，请假起始 {start.date().isoformat()}",
            todo=record,
        )
