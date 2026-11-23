"""声明式基类，单独拆出以避免 models 内部循环依赖。"""
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
