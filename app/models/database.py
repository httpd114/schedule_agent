"""数据库引擎、Session 工厂与初始化。"""
from __future__ import annotations

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from config import ensure_runtime_dirs, settings
from app.models.base import Base

ensure_runtime_dirs()

_connect_args = (
    {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
)
engine = create_engine(settings.database_url, connect_args=_connect_args, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    """建表 + 轻量列迁移（幂等）。"""
    from app.models import entities  # noqa: F401  注册映射

    Base.metadata.create_all(engine)

    # Bug6：旧库可能没有 is_deleted 列，补一次 ALTER TABLE；有则跳过。
    inspector = inspect(engine)
    if "todos" in inspector.get_table_names():
        columns = {item["name"] for item in inspector.get_columns("todos")}
        if "is_deleted" not in columns:
            with engine.begin() as connection:
                connection.execute(text(
                    "ALTER TABLE todos ADD COLUMN is_deleted BOOLEAN NOT NULL DEFAULT 0"
                ))


def get_session():
    """供工具层使用的短会话上下文。"""
    return SessionLocal()
