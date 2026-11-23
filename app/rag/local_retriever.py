"""本地 BM25 检索器：零外部依赖，适合离线/调试场景。"""
from __future__ import annotations

from collections import Counter

import jieba

from app.rag.base import Document, Retriever


class LocalRetriever(Retriever):
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: list[Document] = []
        self.tokens: list[set[str]] = []
        self.tfs: list[Counter] = []
        self.doc_lengths: list[int] = []
        self.avgdl = 0.0
        self.df: Counter = Counter()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t.strip() for t in jieba.lcut(text) if len(t.strip()) > 1]

    def build(self, documents: list[Document]) -> None:
        self.documents = documents
        self.tokens = []
        self.tfs = []
        self.doc_lengths = []
        self.df = Counter()
        for doc in documents:
            tokens = self._tokenize(f"{doc.title} {doc.content}")
            self.tokens.append(set(tokens))
            tf = Counter(tokens)
            self.tfs.append(tf)
            self.doc_lengths.append(len(tokens))
            self.df.update(set(tokens))
        self.avgdl = sum(self.doc_lengths) / len(documents) if documents else 0.0

    def retrieve(self, query: str, top_k: int = 3) -> list[Document]:
        if not self.documents:
            return []
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        n = len(self.documents)
        raw_scores: list[float] = []
        for idx in range(n):
            score = 0.0
            tf = self.tfs[idx]
            dl = self.doc_lengths[idx]
            for token in query_tokens:
                if token not in self.tokens[idx]:
                    continue
                idf = 1.0 + ((n - self.df[token] + 0.5) / (self.df[token] + 0.5))
                freq = tf[token]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                score += idf * (freq * (self.k1 + 1)) / denom
            raw_scores.append(score)

        max_score = max(raw_scores) or 1.0
        ranked = sorted(
            zip(self.documents, raw_scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        results: list[Document] = []
        for doc, score in ranked:
            normalized = score / max_score
            if normalized <= 0:
                continue
            doc.score = round(normalized, 4)
            results.append(doc)
            if len(results) >= top_k:
                break
        return results
