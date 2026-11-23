"""从制度文档抽取出的结构化业务规则。

规则拆成两套集合：
- hard_constraints：本次操作生效，参与 Evaluator 打分与约束校验
- global_baseline_rules：常态化制度（每周一 10 点周会、周五更新看板），
  只作为回答里的附加提示，不参与打分、不算违规。
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class BusinessRule(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    rule_id: str = Field(description="规则标识，如 annual_leave_notice")
    type: Literal["constraint", "guideline"] = Field(default="constraint", description="硬约束或建议")
    category: str = Field(default="general", description="meeting / leave / expense / project / general")
    content: str = Field(description="规则原文摘要")
    enforce: str = Field(
        default="",
        description="可执行校验键：min_lead_hours / work_hours / room_format / min_notice_days / minutes_deadline / approval_flow / expense_window",
    )
    params: dict[str, Any] = Field(default_factory=dict, description="校验参数")
    valid: bool = Field(default=True, description="规则是否有效")
    scope: Literal["hard", "baseline"] = Field(
        default="hard", description="hard=本次hard_constraints；baseline=常态化基线，仅供参考"
    )


# 常态化基线关键词： recurring / 例行安排，不能当成本次冲突
_BASELINE_MARKERS = (
    "每周一", "每周二", "每周三", "每周四", "每周五", "每周六", "每周日",
    "每周", "周会机制", "例会", "更新进度看板", "进度看板", "常态化", "周期性", "日常",
)

# 硬约束关键词：本次操作的资源互斥、时效、格式、审批
_HARD_MARKERS = (
    "会议室", "纪要", "请假", "年假", "报销", "审批", "截止", "提前", "时限",
    "工作时段", "工作时间", "里程碑", "格式", "冲突", "认证", "发票", "销假",
)

_BASELINE_IDS = {
    "weekly_meeting", "weekly_sync", "weekly_status", "friday_board_update",
    "project_weekly_meeting", "weekly_kanban_update",
}


def classify_scope(rule: BusinessRule) -> str:
    """把规则判为 hard 或 baseline。关键词判定是确定性的，不依赖 LLM 心情。"""
    text = f"{rule.rule_id} {rule.content} {rule.category}"
    if rule.rule_id in _BASELINE_IDS or any(word in text for word in _BASELINE_MARKERS):
        return "baseline"
    if any(word in text for word in _HARD_MARKERS):
        return "hard"
    return "hard" if rule.type == "constraint" else "baseline"


def split_rules(items: list[dict]) -> tuple[list[dict], list[dict]]:
    """输入规则 dict 列表，输出 (hard_constraints, global_baseline_rules)。"""
    hard: list[dict] = []
    baseline: list[dict] = []
    for item in items:
        rule = BusinessRule(**item)
        payload = rule.model_dump()
        payload["scope"] = classify_scope(rule)
        (baseline if payload["scope"] == "baseline" else hard).append(payload)
    return hard, baseline


def compact_rules(rules: list[BusinessRule]) -> str:
    """给 Planner / Evaluator 的紧凑硬约束视图。"""
    if not rules:
        return "（无结构化硬约束）"
    lines = []
    for rule in rules:
        if not rule.valid or rule.scope != "hard":
            continue
        suffix = f" [enforce={rule.enforce} {rule.params}]" if rule.enforce else ""
        lines.append(f"- {rule.rule_id}（{rule.category}/硬约束）：{rule.content}{suffix}")
    return "\n".join(lines) or "（无结构化硬约束）"


def compact_baseline(rules: list[BusinessRule]) -> str:
    """常态化基线只做参考提示，明确告知模型不得据此判违规。"""
    if not rules:
        return ""
    lines = [
        f"- {rule.rule_id}：{rule.content}"
        for rule in rules
        if rule.valid and rule.scope == "baseline"
    ]
    if not lines:
        return ""
    return "以下是常态化基线，只可作为回答中的附加提示，**不得据此判定本次方案违规或扣分**：\n" + "\n".join(lines)
