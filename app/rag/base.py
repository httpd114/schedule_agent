"""检索器抽象：本地关键词检索与向量库检索实现同一接口，配置即可切换。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Document:
    id: str
    title: str
    content: str
    source: str = ""
    metadata: dict = field(default_factory=dict)
    score: float = 0.0


class Retriever(ABC):
    @abstractmethod
    def build(self, documents: list[Document]) -> None:
        """建立索引。"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> list[Document]:
        """返回按相关度降序排列的文档片段，score 归一到 0~1。"""
