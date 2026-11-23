"""LangGraph 规划型工作流装配（含 HITL 断点与工具重试降级）。

intent → context →(按需) rag → rule_extractor → planner
  ├─ query → respond → END
  └─ 其它 → evaluator
        ├─ 冲突 → conflict_resolve → evaluator
        ├─ 高风险 → confirm(HITL) → approve→execute / reject→planner
        └─ 低风险可行 → execute
              ├─ 工具失败 → retry(最多2次) → execute / degraded_confirm(HITL 二次强确认)
              └─ 成功 → respond → END
"""
from __future__ import annotations

import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command

import app.tools  # noqa: F401  触发工具注册
from app.graph.nodes import (
    admin_node,
    confirm_node,
    conflict_resolve_node,
    context_node,
    degraded_confirm_node,
    evaluator_node,
    execute_node,
    intent_node,
    planner_node,
    rag_node,
    respond_node,
    retry_node,
    rule_extractor_node,
)
from app.graph.state import AgentState
from app.utils.nl_time import TARGET_KEYS
from config import settings


def _route(state: AgentState) -> str:
    return state.get("route", "respond")


def _intent_route(state: AgentState) -> str:
    """管理员/测试预置指令走独立分支，完全跳过时间解析与工具 Schema。"""
    return "admin" if (state.get("user_intent") or {}).get("category") == "admin" else "context"


def _rag_route(state: AgentState) -> str:
    """RAG 按需触发：查询类默认跳过；制度问题与创建/变更类才检索。"""
    return "rag" if state.get("needs_rag") else "rule_extractor"


def build_workflow():
    workflow = StateGraph(AgentState)
    for name, node in (
        ("intent", intent_node),
        ("admin", admin_node),
        ("context", context_node),
        ("rag", rag_node),
        ("rule_extractor", rule_extractor_node),
        ("planner", planner_node),
        ("evaluator", evaluator_node),
        ("conflict_resolve", conflict_resolve_node),
        ("confirm", confirm_node),
        ("execute", execute_node),
        ("retry", retry_node),
        ("degraded_confirm", degraded_confirm_node),
        ("respond", respond_node),
    ):
        workflow.add_node(name, node)

    workflow.set_entry_point("intent")
    workflow.add_conditional_edges("intent", _intent_route, {"admin": "admin", "context": "context"})
    workflow.add_edge("admin", END)
    workflow.add_conditional_edges("context", _rag_route, {"rag": "rag", "rule_extractor": "rule_extractor"})
    workflow.add_edge("rag", "rule_extractor")
    workflow.add_edge("rule_extractor", "planner")
    workflow.add_conditional_edges("planner", _route, {"evaluate": "evaluator", "respond": "respond"})
    workflow.add_conditional_edges(
        "evaluator",
        _route,
        {"conflict": "conflict_resolve", "confirm": "confirm", "execute": "execute"},
    )
    workflow.add_conditional_edges(
        "conflict_resolve", _route, {"evaluate": "evaluator", "confirm": "confirm"}
    )
    workflow.add_conditional_edges("confirm", _route, {"execute": "execute", "planner": "planner"})
    workflow.add_conditional_edges(
        "execute", _route, {"respond": "respond", "retry": "retry"}
    )
    workflow.add_conditional_edges(
        "retry", _route, {"execute": "execute", "retry": "retry", "degraded_confirm": "degraded_confirm"}
    )
    workflow.add_conditional_edges(
        "degraded_confirm", _route, {"execute": "execute", "respond": "respond"}
    )
    workflow.add_edge("respond", END)
    return workflow


_checkpointer = MemorySaver()
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_workflow().compile(checkpointer=_checkpointer)
    return _graph


def _initial_state(user_input: str, history: list[dict] | None) -> dict:
    return {
        "user_input": user_input,
        "user_intent": {
            "raw": user_input,
            "category": "query",
            "intent_type": "query_schedule",
            "summary": "",
            "time_expressions": {},
            "risk_level": "low",
            "needs_rag": False,
        },
        "context_window": {"start": None, "end": None},
        "target_time": {key: None for key in TARGET_KEYS},
        "context_bundle": {
            "window": {},
            "schedules": [],
            "todos": [],
            "pending_todos": [],
            "near_deadline_todos": [],
            "documents": [],
            "rules": [],
            "hard_constraints": [],
            "global_baseline_rules": [],
            "admin_notes": [],
            "stats": {},
        },
        "candidate_plans": [],
        "selected_plan": None,
        "plan_decision": "pending",
        "alternatives": [],
        "tool_traces": [],
        "feedback": [],
        "messages": [
            {"role": item["role"], "content": item["content"]}
            for item in (history or [])[-10:]
            if item.get("role") in {"user", "assistant"} and item.get("content")
        ],
        "iteration": 0,
        "conflict_rounds": 0,
        "final_answer": "",
        "route": "",
        "rag_documents": [],
        "needs_rag": False,
        "pending_steps": [],
        "retry_attempt": 0,
        "degraded": False,
        "degraded_question": "",
        "degraded_confirmed": False,
    }


def run_agent(
    user_input: str,
    history: list[dict] | None = None,
    thread_id: str | None = None,
    decision: dict | str | None = None,
    include_debug: bool = False,
) -> dict:
    """统一入口。decision 非空时表示恢复被 HITL 暂停的会话。

    默认只返回面向用户的数据；include_debug=True 才附带工具 trace、
    文档权重、打分明细等调试信息。
    """
    graph = get_graph()
    if decision is not None:
        active_thread = thread_id or ""
        result = graph.invoke(
            Command(resume=decision),
            config={
                "configurable": {"thread_id": active_thread},
                "recursion_limit": settings.max_iterations * 4 + 20,
            },
        )
    else:
        active_thread = str(uuid.uuid4())
        result = graph.invoke(
            _initial_state(user_input, history),
            config={
                "configurable": {"thread_id": active_thread},
                "recursion_limit": settings.max_iterations * 4 + 20,
            },
        )

    interruptions = result.get("__interrupt__")
    if interruptions:
        payload = interruptions[0].value if hasattr(interruptions[0], "value") else interruptions[0]
        payload = payload if isinstance(payload, dict) else {"question": str(payload)}
        response = {
            "status": "awaiting_confirmation",
            "kind": payload.get("kind", "plan_confirm"),
            "thread_id": active_thread,
            "question": payload.get("question", "请确认方案"),
            "plans": payload.get("plans", []),
            "selected_plan_id": payload.get("selected_plan_id"),
            "summary": payload.get("summary", ""),
            "baseline_notes": payload.get("baseline_notes", []),
        }
        if payload.get("kind") == "degraded_confirm":
            response.update(
                warning=payload.get("warning", ""),
                pending_steps=payload.get("pending_steps", []),
            )
        return response

    selected = result.get("selected_plan") or {}
    response = {
        "status": "done",
        "thread_id": active_thread,
        "answer": result.get("final_answer", ""),
        "iterations": result.get("iteration", 0),
        "selected_plan": {
            "id": selected.get("id"),
            "title": selected.get("title"),
            "cost": selected.get("cost"),
            "recommendation": selected.get("recommendation"),
        } if selected else None,
        "plan_decision": result.get("plan_decision", "pending"),
    }
    if include_debug:
        from app.graph.nodes import _public_plans

        response["debug"] = {
            "tool_calls": result.get("tool_traces", []),
            "rag_documents": result.get("rag_documents", []),
            "candidate_plans": _public_plans(result.get("candidate_plans", [])),
            "feedback": result.get("feedback", []),
            "alternatives": result.get("alternatives", []),
            "context_stats": (result.get("context_bundle") or {}).get("stats", {}),
            "needs_rag": result.get("needs_rag", False),
            "intent_type": (result.get("user_intent") or {}).get("intent_type"),
            "time_expressions": (result.get("user_intent") or {}).get("time_expressions", {}),
            "context_window": result.get("context_window", {}),
            "target_time": result.get("target_time", {}),
        }
    return response
