"""全局配置：使用 pydantic-settings 从 .env / 环境变量读取。"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """运行期配置。所有字段均可通过环境变量或项目根目录 .env 覆盖。"""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- 智谱 API ----
    zhipu_api_key: str = ""
    zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    zhipu_model: str = "glm-4-plus"
    zhipu_embedding_model: str = "embedding-3"
    request_timeout: float = 60.0

    # ---- 编排策略 ----
    max_iterations: int = Field(default=8, ge=1, le=30)
    deduplicate_tool_calls: bool = True
    # 临近截止阈值：全局待办 deadline 距当前时间 ≤ 该小时数才算“临近截止”。
    # 该判断与用户指定的查询窗口完全解耦，不受窗口影响。
    near_deadline_hours: int = Field(default=24, ge=1, le=720)

    # ---- 持久化 ----
    database_url: str = Field(
        default_factory=lambda: f"sqlite:///{BASE_DIR / 'data' / 'schedule_agent.db'}"
    )

    # ---- RAG ----
    rag_retriever: str = Field(default="local", pattern="^(local|vector)$")
    rag_top_k: int = Field(default=3, ge=1, le=10)
    rag_score_threshold: float = 0.05
    documents_dir: Path = Field(default_factory=lambda: BASE_DIR / "data" / "documents")
    vector_index_dir: Path = Field(default_factory=lambda: BASE_DIR / "data" / "vector_index")

    @property
    def data_dir(self) -> Path:
        return BASE_DIR / "data"


settings = Settings()


def ensure_runtime_dirs() -> None:
    """确保 SQLite 与索引文件所在目录存在。"""
    for path in (settings.data_dir, settings.documents_dir, settings.vector_index_dir):
        Path(path).mkdir(parents=True, exist_ok=True)
