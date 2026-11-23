"""日程 / 待办的 CRUD 仓储，工具层只依赖本模块，不直接接触 ORM Session。"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.database import SessionLocal
from app.models.entities import Schedule, Todo


def _valid_title(title: str) -> str:
    text = str(title or "").strip()
    if len(text) < 2 or len(text) > 200:
        raise ValueError("日程标题需为 2-200 个字符，拒绝空白或单字符脏数据")
    return text


def create_schedule(*, title: str, start_time: datetime, end_time: datetime | None = None, **kwargs) -> dict:
    title = _valid_title(title)
    if not isinstance(start_time, datetime):
        raise ValueError("start_time 必须是 datetime")
    if end_time is not None and end_time <= start_time:
        raise ValueError("end_time 必须晚于 start_time")
    with SessionLocal() as session:
        item = Schedule(title=title, start_time=start_time, end_time=end_time, **kwargs)
        session.add(item)
        session.commit()
        session.refresh(item)
        return item.to_dict()


def list_schedules(
    *,
    keyword: str | None = None,
    start_from: datetime | None = None,
    start_to: datetime | None = None,
    limit: int = 20,
    include_cancelled: bool = False,
) -> list[dict]:
    with SessionLocal() as session:
        stmt = select(Schedule)
        if not include_cancelled:
            stmt = stmt.where(Schedule.status != "cancelled")
        if keyword:
            stmt = stmt.where(Schedule.title.contains(keyword))
        if start_from:
            stmt = stmt.where(Schedule.start_time >= start_from)
        if start_to:
            stmt = stmt.where(Schedule.start_time <= start_to)
        stmt = stmt.order_by(Schedule.start_time.asc()).limit(limit)
        return [row.to_dict() for row in session.scalars(stmt).all()]


def get_schedule(schedule_id: int) -> dict | None:
    with SessionLocal() as session:
        item = session.get(Schedule, int(schedule_id))
        return item.to_dict() if item else None


def delete_schedule(schedule_id: int, reason: str | None = None) -> dict:
    """软删除：状态改为 cancelled，保留记录便于回溯。"""
    with SessionLocal() as session:
        item = session.get(Schedule, int(schedule_id))
        if item is None:
            raise ValueError(f"日程 #{schedule_id} 不存在")
        item.status = "cancelled"
        note = f"[取消] {reason}" if reason else "[取消]"
        item.description = f"{item.description or ''}\n{note}".strip()
        session.commit()
        session.refresh(item)
        return item.to_dict()


def reschedule_schedule(
    original_id: int,
    *,
    title: str,
    start_time: datetime,
    end_time: datetime | None = None,
    **kwargs,
) -> dict:
    """原子改期：同一事务内新建日程 + 软删除原日程。"""
    title = _valid_title(title)
    with SessionLocal() as session:
        original = session.get(Schedule, int(original_id))
        if original is None:
            raise ValueError(f"原日程 #{original_id} 不存在")
        if original.status == "cancelled":
            raise ValueError(f"原日程 #{original_id} 已取消，不能重复改期")
        new_item = Schedule(
            title=title, start_time=start_time, end_time=end_time,
            description=kwargs.get("description") or original.description,
            location=kwargs.get("location") or original.location,
            attendees=kwargs.get("attendees") or original.attendees,
            status="scheduled",
        )
        session.add(new_item)
        original.status = "cancelled"
        original.description = f"{original.description or ''}\n[已改期，新日程见后续记录]".strip()
        session.commit()
        session.refresh(new_item)
        session.refresh(original)
        return {"new": new_item.to_dict(), "original": original.to_dict()}


def purge_invalid_schedules() -> int:
    """把历史脏日程（空白/单字符标题）标记为 cancelled，避免污染聚合与冲突校验。"""
    cleaned = 0
    with SessionLocal() as session:
        for row in session.scalars(select(Schedule)).all():
            title = str(row.title or "").strip()
            if len(title) < 2:
                row.status = "cancelled"
                cleaned += 1
        if cleaned:
            session.commit()
    return cleaned


def create_todo(*, title: str, **kwargs) -> dict:
    with SessionLocal() as session:
        item = Todo(title=title, **kwargs)
        session.add(item)
        session.commit()
        session.refresh(item)
        return item.to_dict()


def list_todos(
    *,
    keyword: str | None = None,
    status: str | None = None,
    due_before: datetime | None = None,
    due_after: datetime | None = None,
    limit: int = 20,
) -> list[dict]:
    with SessionLocal() as session:
        stmt = select(Todo)
        if keyword:
            stmt = stmt.where(Todo.title.contains(keyword))
        if status:
            stmt = stmt.where(Todo.status == status)
        if due_before:
            stmt = stmt.where(Todo.due_time <= due_before)
        if due_after:
            stmt = stmt.where(Todo.due_time >= due_after)
        stmt = stmt.order_by(Todo.due_time.asc().nullslast(), Todo.id.desc()).limit(limit)
        return [row.to_dict() for row in session.scalars(stmt).all()]
