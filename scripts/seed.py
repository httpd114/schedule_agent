"""写入演示数据（按标题去重，可重复执行）。"""
from __future__ import annotations

from datetime import datetime, timedelta

from app.models import repository
from app.models.database import init_db

SCHEDULE_SPECS = [
    {
        "title": "Q4 项目周会",
        "offset_days": 1,
        "hour": 10,
        "duration": 1,
        "location": "12F-A301",
        "attendees": "张伟, 李娜, 王强",
        "description": "同步项目进度与阻塞项",
    },
    {
        "title": "需求评审会",
        "offset_days": 3,
        "hour": 14,
        "minute": 30,
        "duration": 2,
        "location": "10F-B102",
    },
]

TODO_SPECS = [
    {"title": "提交本周费用报销", "offset_days": 2, "hour": 18, "priority": "high", "tags": "财务,报销"},
    {"title": "更新项目进度看板", "offset_days": 1, "hour": 17, "priority": "medium", "tags": "项目"},
]


def seed() -> None:
    init_db()
    base = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    created = 0

    existing_schedules = {item["title"] for item in repository.list_schedules(limit=100)}
    for spec in SCHEDULE_SPECS:
        if spec["title"] in existing_schedules:
            continue
        start = base + timedelta(days=spec["offset_days"], hours=spec["hour"], minutes=spec.get("minute", 0))
        repository.create_schedule(
            title=spec["title"],
            start_time=start,
            end_time=start + timedelta(hours=spec["duration"]),
            location=spec.get("location"),
            attendees=spec.get("attendees"),
            description=spec.get("description"),
        )
        created += 1

    existing_todos = {item["title"] for item in repository.list_todos(limit=100)}
    for spec in TODO_SPECS:
        if spec["title"] in existing_todos:
            continue
        repository.create_todo(
            title=spec["title"],
            due_time=base + timedelta(days=spec["offset_days"], hours=spec["hour"]),
            priority=spec["priority"],
            tags=spec["tags"],
        )
        created += 1

    print(f"[seed] 完成，新增 {created} 条记录（已存在的自动跳过）。")


if __name__ == "__main__":
    seed()
