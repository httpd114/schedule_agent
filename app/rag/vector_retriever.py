"""向量库检索器：智谱 embedding-3 + FAISS，索引持久化到 data/vector_index。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from openai import OpenAI

from config import settings
from app.rag.base import Document, Retriever
from app.rag.local_retriever import LocalRetriever


class VectorRetriever(Retriever):
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.zhipu_api_key,
            base_url=settings.zhipu_base_url,
            timeout=settings.request_timeout,
        )
        self.index_dir = Path(settings.vector_index_dir)
        self.index_path = self.index_dir / "faiss.index"
        self.docs_path = self.index_dir / "documents.json"
        self.index = None
        self.documents: list[Document] = []

    def _embed(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(
            model=settings.zhipu_embedding_model, input=texts
        )
        vectors = np.array([item.embedding for item in resp.data], dtype="float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-9)

    def build(self, documents: list[Document]) -> None:
        """优先复用磁盘索引；不存在则生成并持久化。失败时自动降级为本地检索。"""
        import faiss

        self.documents = documents
        if self.index_path.exists() and self.docs_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                payload = json.loads(self.docs_path.read_text(encoding="utf-8"))
                self.documents = [
                    Document(**{**item, "metadata": item.get("metadata", {})})
                    for item in payload
                ]
                return
            except Exception:
                pass

        if not documents:
            return
        vectors = self._embed([f"{d.title}\n{d.content}" for d in documents])
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        faiss.write_index(self.index, str(self.index_path))
        self.docs_path.write_text(
            json.dumps([d.__dict__ for d in documents], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def retrieve(self, query: str, top_k: int = 3) -> list[Document]:
        if self.index is None or not self.documents:
            return []
        query_vector = self._embed([query])
        scores, indices = self.index.search(query_vector, min(top_k, len(self.documents)))
        results: list[Document] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = self.documents[int(idx)]
            doc.score = round(float(score), 4)
            results.append(doc)
        return results


def build_retriever() -> Retriever:
    """按配置构建检索器；向量方案失败时回退到本地 BM25。"""
    from app.rag.pipeline import load_documents

    documents = load_documents(settings.documents_dir)
    if settings.rag_retriever == "vector":
        try:
            retriever = VectorRetriever()
            retriever.build(documents)
            return retriever
        except Exception as exc:  # noqa: BLE001
            print(f"[RAG] 向量检索初始化失败，降级为本地检索: {exc}")
    local = LocalRetriever()
    local.build(documents)
    return local
