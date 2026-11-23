"""意图、候选方案与规则抽取的 Pydantic 模型。"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.planning.rules import BusinessRule


class ActionType(str, Enum):
    """Bug6：动作类型枚举，Intent 节点最前面识别，删除动作绝不进 create_todo 链路。"""

    QUERY = "query"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"

IntentType = Literal[
    "query_schedule", "query_todo", "query_leave",
    "create_schedule", "create_todo", "apply_leave", "check_room",
    "reschedule", "policy", "admin", "other",
]


class ModelBase(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class IntentModel(ModelBase):
    category: Literal["query", "create", "change", "plan", "admin"] = Field(
        description="query=纯查询；create=新建；change=变更已有安排；plan=多方案推演；admin=管理员/测试预置指令"
    )
    intent_type: IntentType = Field(
        default="other",
        description="语义子类型，决定时间原文路由到哪个提取器",
    )
    summary: str = Field(description="对用户意图的一句话概括")
    action_type: ActionType = Field(
        default=ActionType.CREATE,
        description="动作类型：query/create/update/delete；删除类由 Intent 节点确定性识别",
    )
    target: Literal["todo", "schedule", "none"] = Field(
        default="none", description="动作作用对象：delete 时必填 todo 或 schedule"
    )
    target_id: int | None = Field(default=None, description="delete 动作的目标记录 ID")
    time_expressions: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "按语义键给出的用户时间表达原文，例如 schedule_start / schedule_end / "
            "todo_deadline / leave_date / check_time / context_start / context_end。"
            "这些原文只交给 parse_nl_time，绝不直接进工具 Schema。"
        ),
    )
    risk_level: Literal["low", "high"] = Field(default="low", description="是否属于高风险变更")
    needs_rag: bool = Field(
        default=False,
        description="是否需要检索制度文档：查询类默认 false，主动问制度或创建/变更规划类为 true",
    )
    # 兼容旧版 LLM 输出；保留字段但内部流转不再使用
    time_hint: str | None = Field(default=None, description="[已废弃] 旧版通用时间原文")
    window_start: str | None = Field(default=None, description="[已废弃] 旧版窗口起点")
    window_end: str | None = Field(default=None, description="[已废弃] 旧版窗口终点")


class PlanStepModel(ModelBase):
    tool: Literal[
        "create_schedule", "query_schedules", "create_todo", "query_todos",
        "search_documents", "delete_schedule", "delete_todo", "reschedule_schedule",
        "apply_leave", "check_room_occupancy",
    ] = Field(description="要调用的工具名")
    args: dict[str, Any] = Field(default_factory=dict, description="工具参数对象")


class CandidatePlanModel(ModelBase):
    id: str = Field(description="方案编号，如 A / B / C")
    title: str = Field(description="方案标题")
    description: str = Field(default="", description="方案说明")
    steps: list[PlanStepModel] = Field(default_factory=list, description="有序工具调用步骤")
    risks: list[str] = Field(default_factory=list, description="潜在风险")
    cost: Literal["low", "medium", "high"] = Field(default="low", description="执行代价")
    start_time: str | None = Field(default=None, description="方案涉及时间的起点 ISO 8601")
    end_time: str | None = Field(default=None, description="方案涉及时间的终点 ISO 8601")
    needs_confirmation: bool = Field(default=False, description="是否需要用户确认后执行")
    reschedule_of: int | None = Field(
        default=None, description="改期方案的原日程 ID：新建与软删除原日程在同一事务完成"
    )
    recommendation: str | None = Field(default=None, description="给用户看的 concret 推荐文案")


class PlanSet(ModelBase):
    plans: list[CandidatePlanModel] = Field(description="1~3 个候选执行方案")


class RuleSet(ModelBase):
    hard_constraints: list[BusinessRule] = Field(
        default_factory=list, description="本次操作生效的硬约束，参与打分校验"
    )
    global_baseline_rules: list[BusinessRule] = Field(
        default_factory=list, description="常态化制度，只做参考提示，不计违规不计分"
    )
    rules: list[BusinessRule] = Field(default_factory=list, description="兼容旧输出的合并规则")
