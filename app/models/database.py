"""数据库引擎、Session 工厂与初始化。"""
from __future__ import annotations

from sqlalchemy import create_engine
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
    """建表（幂等）。"""
    from app.models import entities  # noqa: F401  注册映射

    Base.metadata.create_all(engine)


def get_session():
    """供工具层使用的短会话上下文。"""
    return SessionLocal()
