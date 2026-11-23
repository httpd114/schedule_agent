"""RAG 流水线：文档加载/切分、检索缓存、要点合并。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from config import settings
from app.rag.base import Document, Retriever


def load_documents(directory: Path | str) -> list[Document]:
    """加载 .md/.txt 业务文档并按段落切块。"""
    documents: list[Document] = []
    root = Path(directory)
    if not root.exists():
        return documents

    for path in sorted(root.glob("*")):
        if path.suffix.lower() not in {".md", ".txt"} or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        title_line = next(
            (line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("# ")),
            path.stem,
        )
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[str] = []
        buffer = ""
        for paragraph in paragraphs:
            candidate = f"{buffer}\n\n{paragraph}".strip()
            if len(candidate) > 600 and buffer:
                chunks.append(buffer)
                buffer = paragraph
            else:
                buffer = candidate
        if buffer:
            chunks.append(buffer)

        for idx, chunk in enumerate(chunks):
            documents.append(
                Document(
                    id=f"{path.stem}-{idx}",
                    title=title_line if idx == 0 else f"{title_line}（{idx + 1}）",
                    content=chunk,
                    source=path.name,
                    metadata={"path": str(path)},
                )
            )
    return documents


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """进程内单例，避免每次会话重复建索引。"""
    from app.rag.vector_retriever import build_retriever

    return build_retriever()


def retrieve(query: str, top_k: int | None = None, threshold: float | None = None) -> list[Document]:
    docs = get_retriever().retrieve(query, top_k or settings.rag_top_k)
    cutoff = settings.rag_score_threshold if threshold is None else threshold
    return [doc for doc in docs if doc.score >= cutoff]


def format_context(documents: list[Document]) -> str:
    parts = []
    for idx, doc in enumerate(documents, 1):
        parts.append(f"[{idx}] {doc.title}（来源：{doc.source}，相关度 {doc.score:.2f}）\n{doc.content}")
    return "\n\n".join(parts)


def merge_answer(llm, user_input: str, answer: str, documents: list[Document]) -> str:
    """把检索到的业务要点合并进最终回答。检索结果为空时原样返回。"""
    if not documents:
        return answer
    prompt = f"""你是回答润色器。请把参考资料中的要点自然合并进「原回答」，输出一份完整、简洁的最终回答。

要求：
1. 不要臆造资料中没有的信息；若资料与原回答冲突，以资料为准并简要说明。
2. 保留原回答中已经完成的操作结论（如已创建的日程/待办编号）。
3. 用 Markdown 输出，不要提及“参考资料”“原回答”等元信息。

用户问题：{user_input}

原回答：
{answer}

参考资料：
{format_context(documents)}
"""
    try:
        message = llm.chat(
            [
                {"role": "system", "content": "你是一个严谨的办公助理。"},
                {"role": "user", "content": prompt},
            ],
            tools=None,
        )
        content = (message.get("content") or "").strip()
        return content or answer
    except Exception as exc:  # noqa: BLE001 - RAG 增强失败不应破坏主流程
        print(f"[RAG] 要点合并失败，保留原回答: {exc}")
        return answer
