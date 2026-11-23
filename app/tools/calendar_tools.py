"""日程类工具：创建日程、查询日程、查询会议室占用。入参 Schema 见 app/tools/schemas.py。"""
from __future__ import annotations

from datetime import timedelta

from app.models import repository
from app.tools.base import BaseTool, ToolExecutionError, success_payload
from app.tools.registry import register_tool
from app.tools.schemas import (
    CheckRoomOccupancyArgs,
    CreateScheduleArgs,
    DeleteScheduleArgs,
    QuerySchedulesArgs,
    RescheduleScheduleArgs,
)
from app.utils.validators import parse_datetime


@register_tool
class CreateScheduleTool(BaseTool):
    name = "create_schedule"
    description = (
        "创建一条日程/会议安排。用于用户说'安排/约/新增日程'时。"
        "start_time/end_time 必须是 ISO 8601（由时间解析层生成）；"
        "room_prefer 为用户指定或求解器推荐的会议室。"
    )
    args_model = CreateScheduleArgs

    def execute(self, *, title, start_time, end_time=None, room_prefer=None, description=None, attendees=None, **_) -> str:
        start = parse_datetime(start_time, "start_time")
        end = parse_datetime(end_time, "end_time") if end_time else None
        if end is not None and end <= start:
            raise ToolExecutionError(
                "INVALID_TIME_RANGE",
                f"结束时间 {end_time} 必须晚于开始时间 {start_time}",
                "请重新生成两个时间参数，保证 end_time > start_time。",
            )
        if isinstance(attendees, list):
            attendees = ", ".join(str(a) for a in attendees) or None
        record = repository.create_schedule(
            title=title,
            start_time=start,
            end_time=end,
            location=room_prefer,
            description=description,
            attendees=attendees,
        )
        return success_payload(
            message=f"已创建日程 #{record['id']}：{record['title']}，开始于 {record['start_time']}",
            schedule=record,
        )


@register_tool
class QuerySchedulesTool(BaseTool):
    name = "query_schedules"
    description = (
        "查询 context_window 时间区间内的日程列表。"
        "窗口由意图解析层按业务时间自动生成（前后各一天），不要接收用户原始时间文本。"
    )
    args_model = QuerySchedulesArgs

    def execute(
        self, *, context_window=None, keyword=None, date=None,
        start_from=None, start_to=None, limit=20, **_
    ) -> str:
        start_dt = parse_datetime(start_from, "start_from", default_hour=0) if start_from else None
        end_dt = parse_datetime(start_to, "start_to", default_hour=23) if start_to else None
        if context_window:
            window = context_window if isinstance(context_window, dict) else {}
            if window.get("start"):
                start_dt = parse_datetime(window["start"], "context_window.start", default_hour=0)
            if window.get("end"):
                end_dt = parse_datetime(window["end"], "context_window.end", default_hour=23)
        if date:
            day = parse_datetime(date, "date", default_hour=0)
            start_dt = day.replace(hour=0, minute=0, second=0, microsecond=0)
            end_dt = day.replace(hour=23, minute=59, second=59, microsecond=0)

        records = repository.list_schedules(
            keyword=keyword, start_from=start_dt, start_to=end_dt, limit=limit
        )
        return success_payload(
            message=f"共查询到 {len(records)} 条日程。",
            count=len(records),
            schedules=records,
        )


@register_tool
class CheckRoomOccupancyTool(BaseTool):
    name = "check_room_occupancy"
    description = (
        "查询某个会议室在指定时刻是否被占用。用于用户问'11月26日14点 15F-B205 有没有空'时。"
        "check_datetime 必须是 ISO 8601 单点时间，由时间解析层从用户表达生成。"
    )
    args_model = CheckRoomOccupancyArgs

    def execute(self, *, check_datetime, room_name, **_) -> str:
        point = parse_datetime(check_datetime, "check_datetime")
        day_start = point.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = point.replace(hour=23, minute=59, second=59, microsecond=0)
        candidates = repository.list_schedules(start_from=day_start, start_to=day_end, limit=100)
        occupied = []
        target = str(room_name).strip()
        for item in candidates:
            if str(item.get("location") or "").strip() != target:
                continue
            start = parse_datetime(item["start_time"], "start_time")
            end = (
                parse_datetime(item["end_time"], "end_time")
                if item.get("end_time") else start + timedelta(hours=1)
            )
            if start <= point <= end:
                occupied.append(item)
        return success_payload(
            message=(
                f"{point.isoformat(timespec='seconds')} {target} "
                + ("已被占用" if occupied else "空闲")
            ),
            room=target,
            check_time=point.isoformat(timespec="seconds"),
            occupied=bool(occupied),
            schedules=occupied,
        )


@register_tool
class DeleteScheduleTool(BaseTool):
    name = "delete_schedule"
    description = (
        "取消/删除一条日程（软删除，状态变为 cancelled 可回溯）。"
        "用于改期方案中确认后自动取消原日程，或用户明确要求取消会议时。"
        "必须提供日程 id。"
    )
    args_model = DeleteScheduleArgs

    def execute(self, *, id: int, reason: str | None = None, **_) -> str:
        record = repository.delete_schedule(id, reason=reason)
        return success_payload(
            message=f"已取消日程 #{record['id']}：{record['title']}（软删除，可回溯）",
            schedule=record,
        )


@register_tool
class RescheduleScheduleTool(BaseTool):
    name = "reschedule_schedule"
    description = (
        "改期专用工具：把一条已有日程改到新时间。"
        "在原子事务内新建日程并软删除原日程，可回溯。"
        "schedule_id 为原日程 ID；start_time/end_time 由约束求解器在目标时段内选出；"
        "target_date/period_prefer/room_prefer 记录用户的模糊时段偏好。"
    )
    args_model = RescheduleScheduleArgs

    def execute(
        self, *, schedule_id: int, target_date: str | None = None,
        period_prefer: str | None = None, room_prefer: str | None = None,
        start_time: str | None = None, end_time: str | None = None, **_
    ) -> str:
        if not start_time:
            raise ToolExecutionError(
                "MISSING_TARGET_TIME",
                "改期缺少新开始时间：请先由约束求解器在目标时段内选出具体时段。",
                "调用前应在目标日期的时段区间内用 CSP 求解 start_time/end_time。",
            )
        original = repository.get_schedule(int(schedule_id))
        if original is None:
            raise ToolExecutionError(
                "SCHEDULE_NOT_FOUND", f"原日程 #{schedule_id} 不存在", "请核对日程 ID。"
            )
        if str(original.get("status")) == "cancelled":
            raise ToolExecutionError(
                "SCHEDULE_CANCELLED", f"原日程 #{schedule_id} 已取消，不能重复改期", "请核对日程 ID。"
            )
        start = parse_datetime(start_time, "start_time")
        end = parse_datetime(end_time, "end_time") if end_time else start + timedelta(hours=1)
        if end <= start:
            raise ToolExecutionError(
                "INVALID_TIME_RANGE", "新结束时间必须晚于开始时间", "请重新选择时段。"
            )
        result = repository.reschedule_schedule(
            int(schedule_id),
            title=original["title"],
            start_time=start,
            end_time=end,
            location=room_prefer or original.get("location"),
            description=original.get("description"),
            attendees=original.get("attendees"),
        )
        return success_payload(
            message=(
                f"已将日程 #{schedule_id}「{original['title']}」改到 "
                f"{result['new']['start_time']}"
                + (f"（{period_prefer}，会议室 {room_prefer}）" if period_prefer and room_prefer else "")
                + "；原日程已软删除，可回溯"
            ),
            new_schedule=result["new"],
            original=result["original"],
        )
