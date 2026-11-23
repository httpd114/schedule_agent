"""日程 / 待办 ORM 实体。"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Schedule(Base):
    __tablename__ = "schedules"
    __table_args__ = (Index("ix_schedules_start_time", "start_time"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    location: Mapped[str | None] = mapped_column(String(200), nullable=True)
    attendees: Mapped[str | None] = mapped_column(String(500), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="scheduled", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "start_time": self.start_time.isoformat(timespec="minutes"),
            "end_time": self.end_time.isoformat(timespec="minutes") if self.end_time else None,
            "location": self.location,
            "attendees": self.attendees,
            "status": self.status,
        }


class Todo(Base):
    """待办实体。

    ⚠️ 时间语义硬约束（禁止在其它模块绕过）：
    - 待办的一切【时间筛选 / 排序 / 临近截止告警】，唯一可信源是本实体的 due_time（即 todo.deadline）。
    - 由日程衍生出的纪要/行动项待办，其 due_time 生成公式固定为「关联日程 end_time + 24h」，
      一旦写入即成为该待办自身的时间，与关联日程再无筛选关系。
    - description 里出现的“关联日程 #N”仅用于展示关联信息，
      ❌ 严禁用于待办时间过滤、排序或临近判断，严禁拿父日程 start/end 当待办时间。
    """
    __tablename__ = "todos"
    __table_args__ = (Index("ix_todos_status", "status"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    due_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    priority: Mapped[str] = mapped_column(String(16), default="medium", nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    tags: Mapped[str | None] = mapped_column(String(300), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "due_time": self.due_time.isoformat(timespec="minutes") if self.due_time else None,
            "priority": self.priority,
            "status": self.status,
            "tags": self.tags,
        }
