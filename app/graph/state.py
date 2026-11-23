"""三层结构化全局状态（时间语义已分离）。

user_intent     —— 原始意图、分类与语义子类型
context_window  —— 【唯一】聚合查询窗口 {start, end}，仅供 context_service 拉取区间数据
                    只由 parse_nl_time 从业务时间自动推导，绝不接收用户原始时间串
target_time     —— 业务时间分组：会议起止 / 待办截止 / 请假日期 / 查会议室单点
                    由对应工具的 Pydantic Schema 接收 ISO 8601
context_bundle  —— 聚合上下文（日程、待办、文档、硬约束、常态化基线）
                    ⚠️ 待办时间语义硬约束：
                    - todos 只含 deadline 落在 context_window 内的待办（用于回答“某天的待办”）
                    - pending_todos 为全局未完成待办（仅供延后提示/关联展示）
                    - near_deadline_todos 为全局 deadline 临近的待办（与查询窗口完全解耦）
                    - 三者唯一可信时间源都是 todo.deadline；linked_schedule 仅展示，禁止参与筛选/告警
candidate_plans / selected_plan / plan_decision —— 规划与用户决策
tool_traces / feedback —— 执行轨迹与反馈，仅供调试与自愈

规则分两套：
- hard_constraints     本次操作生效，参与打分/违规校验
- global_baseline_rules 常态化制度，只做参考提示，不计违规、不计分
"""
from __future__ import annotations

from typing import TypedDict


class UserIntent(TypedDict):
    raw: str
    category: str          # query | create | change | plan | admin
    intent_type: str       # query_schedule | create_schedule | apply_leave | check_room ...
    summary: str
    time_expressions: dict[str, str]  # 用户时间表达原文，仅用于展示/再解析，不进 schema
    risk_level: str        # low | high
    needs_rag: bool        # 查询类默认 False，制度问题/创建变更类为 True


class ContextWindow(TypedDict):
    start: str | None      # ISO 8601，程序生成的查询窗口起点
    end: str | None        # ISO 8601，程序生成的查询窗口终点


class TargetTime(TypedDict):
    schedule_start: str | None
    schedule_end: str | None
    todo_deadline: str | None
    leave_date: str | None
    leave_end: str | None
    check_time: str | None


class ContextBundle(TypedDict):
    window: dict
    schedules: list[dict]
    todos: list[dict]                    # 仅 deadline 落在 window 内
    pending_todos: list[dict]            # 全局未完成（延后提示用，不当窗口结果展示）
    near_deadline_todos: list[dict]      # 全局临近截止（独立判断，带距离信息）
    documents: list[dict]
    rules: list[dict]                    # 兼容旧视图：硬约束 + 基线
    hard_constraints: list[dict]         # 本次生效、参与打分
    global_baseline_rules: list[dict]    # 常态化制度、只做提示
    admin_notes: list[str]               # 管理员/测试预置指令，不进时间校验链路
    stats: dict


class AgentState(TypedDict):
    user_input: str
    user_intent: UserIntent
    context_window: ContextWindow
    target_time: TargetTime
    context_bundle: ContextBundle
    candidate_plans: list[dict]
    selected_plan: dict | None
    plan_decision: str            # pending | approved | rejected | modified
    alternatives: list[dict]
    tool_traces: list[dict]
    feedback: list[str]
    messages: list[dict]
    iteration: int
    conflict_rounds: int
    final_answer: str
    route: str
    rag_documents: list[dict]
    needs_rag: bool
    pending_steps: list[dict]     # 执行/重试子图的待处理步骤
    retry_attempt: int            # 单步最多重试 2 次
    degraded: bool                # 工具彻底失败，已进入二次强确认
    degraded_question: str
    degraded_confirmed: bool
