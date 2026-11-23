"""文档检索工具：RAG 入口，作为第五类工具暴露给模型。"""
from __future__ import annotations

from app.rag import pipeline
from app.tools.base import BaseTool, success_payload
from app.tools.registry import register_tool
from app.tools.schemas import SearchDocumentsArgs


@register_tool
class SearchDocumentsTool(BaseTool):
    name = "search_documents"
    description = (
        "检索公司内部业务文档（报销制度、会议规范、项目排期、请假流程等）。"
        "当用户问题涉及制度、流程、规范、文档要点，或需要依据资料回答时调用。"
    )
    args_model = SearchDocumentsArgs

    def execute(self, *, query, top_k=3, **_) -> str:
        documents = pipeline.retrieve(query, top_k=top_k)
        return success_payload(
            message=f"检索到 {len(documents)} 条相关文档片段。",
            count=len(documents),
            documents=[
                {
                    "title": doc.title,
                    "content": doc.content,
                    "source": doc.source,
                    "score": doc.score,
                }
                for doc in documents
            ],
        )
