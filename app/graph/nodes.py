"""规划型工作流节点：意图 → 聚合上下文 → 按需 RAG → 规则抽取 →
模板/LLM 规划 → 加权评估 → 冲突消解 / HITL 确认 → 重试降级 → 工具执行 → 回答。

调试信息（工具 trace、文档权重、打分明细）只留在 State，不进入用户回答。
"""
from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, timedelta

from langgraph.types import interrupt

from app.llm.zhipu_client import get_llm
from app.models import repository
from app.planning.constraint_solver import (
    evaluate_plan,
    format_alternative,
    format_recommendation,
    plan_window,
    suggest_windows,
)
from app.planning.llm_json import parse_llm_json
from app.planning.plan_models import ActionType, IntentModel, PlanSet, RuleSet
from app.planning.rules import BusinessRule, compact_baseline, compact_rules, split_rules
from app.planning.templates import build_templates, template_hint
from app.rag import pipeline
from app.services.context_service import aggregate_context
from app.tools.base import structured_error
from app.tools.registry import call_tool, get_tool
from app.tools.schemas import sanitize_args
from app.utils.nl_time import _parse_period_prefer, parse_nl_time
from app.utils.validators import parse_datetime
from config import settings

# 查询类只有主动问制度/流程时才检索文档
_RAG_KEYWORDS = (
    "制度", "规定", "流程", "规范", "报销", "考勤", "请假规则", "会议室",
    "纪要", "审批", "里程碑", "提前几天", "提前多久", "限额", "额度",
    "临近截止", "临近", "延后", "宽限", "能不能推迟", "时限",
)


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


_ADMIN_KEYWORDS = ("预置", "测试数据", "固定周会", "已被占用", "管理员指令", "系统设置")

# LLM 偶尔会漏提“明天”这类相对时间，这里做确定性扫描兜底，
# 保证查询窗口不依赖模型是否提取时间表达。
_QUERY_TIME_PATTERN = re.compile(
    r"大后天|后天|明天|明日|今天|今日|下下?周[一二三四五六日天]"
    r"|本?周[一二三四五六日天]|\d{1,2}月\d{1,2}[号日]|\d{4}-\d{1,2}-\d{1,2}"
)


# Bug6：删除动作确定性识别关键词。命中“删除/移除/取消 + ID”即判 DELETE，
# 绝不进入 create_todo 链路（否则会生成“删除整理技术文档”这样的新待办）。
_DELETE_KEYWORDS = ("删除", "移除", "取消", "干掉", "删掉")
_TODO_WORDS = ("待办", "任务", "todo", "事项")
_SCHEDULE_WORDS = ("日程", "会议", "安排", "评审", "周会")
# 根治：只认显式 ID:数字 / ID：数字 / 编号:数字，不再用裸数字兜底，
# 否则“11月27日删除待办”会把日期里的 27 误当成 ID。
_DELETE_ID_RE = re.compile(r"(?:ID|id|编号|序号)\s*[:：#]?\s*(\d+)")


def parse_delete_command(user_input: str) -> dict:
    """按用户推荐方案，用独立正则提取删除命令的 ID 与目标类型。

    返回结构：
    - 命中：{"action": ActionType.DELETE, "target_id": int, "target_type": "todo"|"schedule"}
    - 未命中：{"action": None, "target_id": None, "target_type": None}

    完全不依赖 TARGET_KEYS 或任何时间抽取循环，适配带备注的长文本。
    """
    value = str(user_input or "")
    match = _DELETE_ID_RE.search(value)
    if not match:
        return {"action": None, "target_id": None, "target_type": None}
    target_id = int(match.group(1))
    lowered = value.lower()
    if "待办" in value or "todo" in lowered or any(w in value for w in _TODO_WORDS):
        target_type = "todo"
    elif "日程" in value or "会议" in value or any(w in value for w in _SCHEDULE_WORDS):
        target_type = "schedule"
    else:
        # 默认待办：与用户推荐写法一致（“删除整理技术文档 ID:2”没有显式类型词）
        target_type = "todo"
    return {"action": ActionType.DELETE, "target_id": target_id, "target_type": target_type}


def _detect_delete_action(text: str) -> dict | None:
    """Intent 节点最前面调用：命中“删除关键词 + 显式 ID”才判 DELETE。"""
    value = str(text or "")
    if not any(word in value for word in _DELETE_KEYWORDS):
        return None
    # 排除“取消会议改到X / 调整到下午”这类改期语义，避免误伤改期流程
    if any(word in value for word in ("改期", "调整到", "改到", "换到", "移到", "推迟到")):
        return None
    parsed = parse_delete_command(value)
    if parsed["action"] != ActionType.DELETE:
        return None
    return {
        "action_type": "delete",
        "target": parsed["target_type"],
        "target_id": parsed["target_id"],
    }


def _scan_query_time(text: str) -> dict | None:
    """从用户原文扫描查询时间表达，作为 LLM 漏提时的确定性兜底。"""
    match = _QUERY_TIME_PATTERN.search(text or "")
    return {"context_start": match.group(0)} if match else None


def _fallback_intent(text: str) -> IntentModel:
    """LLM 输出不可解析时的确定性兜底：同时给出语义子类型。"""
    category, intent_type = "query", "query_schedule"
    if any(word in text for word in _ADMIN_KEYWORDS):
        category, intent_type = "admin", "admin"
    elif any(word in text for word in ("请假", "休假")):
        category, intent_type = "change", "apply_leave"
    elif any(word in text for word in ("占用", "有空", "空闲", "会议室")):
        category, intent_type = "query", "check_room"
    elif any(word in text for word in ("改期", "调整到", "推迟", "提前到", "取消", "换到")):
        category, intent_type = "change", "reschedule"
    elif any(word in text for word in ("推演", "方案", "怎么安排", "几个选项")):
        category, intent_type = "plan", "create_schedule"
    elif any(word in text for word in ("待办", "提醒我", "记个", "任务")):
        category, intent_type = "create", "create_todo"
    elif any(word in text for word in ("创建", "安排", "新增", "建个")):
        category, intent_type = "create", "create_schedule"
    elif any(word in text for word in ("制度", "规定", "流程", "报销", "考勤")):
        intent_type = "policy"
    return IntentModel(
        category=category,
        intent_type=intent_type,
        summary=text[:120],
        time_expressions={},
        risk_level="high" if category in {"change", "plan"} else "low",
        needs_rag=category != "query" or any(word in text for word in _RAG_KEYWORDS),
    )


def _fallback_plans(intent: dict, user_input: str, target_time: dict | None = None) -> PlanSet:
    """LLM 输出不可解析时的确定性兜底方案。时间只从结构化 target_time 读取。"""
    target = target_time or {}
    hint = target.get("schedule_start") or intent.get("time_hint")
    start = None
    if hint:
        try:
            start = parse_datetime(hint)
        except Exception:  # noqa: BLE001
            start = None
    if start is None:
        start = (datetime.now() + timedelta(days=1)).replace(hour=10, minute=0, second=0, microsecond=0)
    end = start + timedelta(hours=1)
    base_args = {
        "title": intent.get("summary") or user_input[:40],
        "start_time": start.isoformat(timespec="seconds"),
        "end_time": end.isoformat(timespec="seconds"),
    }
    # 兜底方案生成的纪要待办同样带上 deadline = 会议结束 + 24h，
    # 保证待办自身时间可信源存在，不依赖关联日程做筛选。
    minutes_due = (end + timedelta(hours=24)).isoformat(timespec="seconds")
    return PlanSet(
        plans=[
            {
                "id": "A",
                "title": "按指定时间创建日程",
                "description": "直接创建日程并同步记录后续待办",
                "steps": [
                    {"tool": "create_schedule", "args": base_args},
                    {"tool": "create_todo", "args": {"content": "会后整理纪要", "deadline": minutes_due, "priority": "medium"}},
                ],
                "risks": ["未校验与既有日程的冲突"],
                "cost": "low",
                "start_time": base_args["start_time"],
                "end_time": base_args["end_time"],
            },
            {
                "id": "B",
                "title": "仅创建待办，稍后再定时间",
                "description": "不占用时间资源，先记录任务",
                "steps": [{"tool": "create_todo", "args": {"title": base_args["title"], "priority": "medium"}}],
                "risks": ["没有确定日程时间"],
                "cost": "low",
            },
        ]
    )


def _bundle(state: dict) -> dict:
    current = state.get("context_bundle") or {}
    return {
        "window": current.get("window", {}),
        "schedules": current.get("schedules", []),
        "todos": current.get("todos", []),
        "pending_todos": current.get("pending_todos", []),
        "near_deadline_todos": current.get("near_deadline_todos", []),
        "documents": current.get("documents", []),
        "rules": current.get("rules", []),
        "hard_constraints": current.get("hard_constraints", []),
        "global_baseline_rules": current.get("global_baseline_rules", []),
        "admin_notes": current.get("admin_notes", []),
        "stats": current.get("stats", {}),
    }


def _hard_rules(bundle: dict) -> list[BusinessRule]:
    return [BusinessRule(**item) for item in bundle.get("hard_constraints", [])]


def _trace(tool: str, args: dict, ok: bool = True, summary: str = "", result: str = "") -> dict:
    return {"tool": tool, "args": args, "ok": ok, "summary": summary, "result": result}


def _compact(data: list[dict], limit: int = 12) -> str:
    return json.dumps(data[:limit], ensure_ascii=False)[:3000]


def _cost_rank(cost: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(str(cost), 1)


def intent_node(state: dict) -> dict:
    """意图识别 + 按语义分流提取时间：context_window 与 target_time 彻底分离。"""
    user_input = state["user_input"]

    # Bug6：最前面确定性识别删除动作，命中即短路，绝不进入 create_todo 链路。
    delete_action = _detect_delete_action(user_input)
    if delete_action is not None:
        is_schedule = delete_action["target"] == "schedule"
        return {
            "user_intent": {
                "raw": user_input,
                "category": "delete",
                "intent_type": "delete_schedule" if is_schedule else "delete_todo",
                "summary": f"删除{'日程' if is_schedule else '待办'} ID:{delete_action['target_id']}",
                "action_type": "delete",
                "target": delete_action["target"],
                "target_id": delete_action["target_id"],
                "time_expressions": {},
                "risk_level": "high",
                "needs_rag": False,
            },
            "context_window": {"start": None, "end": None},
            # 删除动作不解析任何业务时间，直接给空字典；
            # 根治 NameError：这里绝不再引用未导入的 TARGET_KEYS。
            "target_time": {},
            "time_prefer": None,
            "needs_rag": False,
            "route": "delete",
            "messages": list(state.get("messages", [])) + [{"role": "user", "content": user_input}],
            "plan_decision": "pending",
            "tool_traces": [],
            "feedback": [],
            "alternatives": [],
            "conflict_rounds": 0,
            "iteration": 0,
            "pending_steps": [],
            "retry_attempt": 0,
            "degraded": False,
            "degraded_question": "",
            "degraded_confirmed": False,
        }

    message = get_llm().chat(
        [
            {
                "role": "system",
                "content": f"""你是意图分类器。当前时间 {_now_text()}。输出严格 JSON：
{{"category":"query|create|change|plan|admin","intent_type":"query_schedule|query_todo|query_leave|create_schedule|create_todo|apply_leave|check_room|reschedule|policy|admin","summary":"一句话概括用户想做的事","time_expressions":{{}},"risk_level":"low|high","needs_rag":true/false}}

time_expressions 必须按业务语义选键，值只填用户说的原文，不要自己换算时间：
- 查日程/待办/请假记录：context_start、context_end（用户想看的日期范围）。
  ⚠️ 必须提取“今天/明天/后天/下周一/11月26日”这类时间！
  例如“看看我明天有哪些待办” → time_expressions = {{"context_start": "明天"}}，绝不能留空。
- 创建会议/安排评审/改期：schedule_start、schedule_end。区间可整体放 schedule_start，如"下周四下午14:00–15:30"
- 新增待办：todo_deadline，如"截止到11月26日"
- 请假：leave_date，如"下周一到周三"
- 查会议室占用：check_time，如"11.26 14:00"
没有时间时 time_expressions 为空对象。

关键规则：
- 用户只是在告知测试/预置信息（如"周一10:00固定周会、11.26 14:00 15F-B205已被占用"），category=admin，time_expressions 必须为空对象，不解析任何时间。
- 查询类用 query；新建用 create；修改/取消/请假用 change；需要多方案推演用 plan。
- 涉及批量变更、跨多天请假、影响他人日程时 risk_level=high。
- needs_rag：创建/变更/规划类为 true；查询类仅当主动询问制度、规定、流程、报销、考勤、审批时为 true，普通"查日程/查待办"为 false。
只输出 JSON。""",
            },
            {"role": "user", "content": user_input},
        ],
        tools=None,
    )
    model = parse_llm_json(message["content"], IntentModel, _fallback_intent(user_input))
    parsed = parse_nl_time(model.time_expressions or {}, model.intent_type, now=datetime.now())
    context_window, target_time = parsed.to_state()
    time_prefer = parsed.period_prefer
    # 确定性兜底：LLM 没把模糊时段放进 time_expressions 时，从原文补一次扫描
    if time_prefer is None and model.intent_type in {"create_schedule", "reschedule", "plan"}:
        prefer_hit = _parse_period_prefer(
            user_input, now=datetime.now(), base_date=datetime.now().date(), target_key="schedule_start"
        )
        if prefer_hit is not None:
            time_prefer = prefer_hit

    # 确定性兜底：查询类若 LLM 漏提时间，从原文扫描“明天”等相对时间生成查询窗口。
    # 这一步不依赖模型稳定性，确保“明天有哪些待办”一定按 11-24 窗口筛选。
    if model.category == "query" and not any(context_window.values()):
        scanned = _scan_query_time(user_input)
        if scanned:
            context_window, _ = parse_nl_time(
                scanned, model.intent_type, now=datetime.now()
            ).to_state()
            model.time_expressions = {**(model.time_expressions or {}), **scanned}

    # 兼容旧版 LLM 输出：这些字段一律当作“原文”重新解析，绝不直接透传给 schema
    if not any(context_window.values()) and model.time_hint:
        context_window, target_time = parse_nl_time(
            model.time_hint, model.intent_type, now=datetime.now()
        ).to_state()
    if not context_window.get("start") and model.window_start and model.window_end:
        legacy_window = parse_nl_time(
            {"context_start": model.window_start, "context_end": model.window_end},
            model.intent_type, now=datetime.now(),
        ).to_state()[0]
        if legacy_window.get("start"):
            context_window = legacy_window

    keyword_hit = any(word in user_input for word in _RAG_KEYWORDS)
    needs_rag = keyword_hit or model.category in {"create", "change", "plan"} or bool(model.needs_rag and keyword_hit)
    is_admin = model.category == "admin"
    if is_admin:
        needs_rag = False
    intent = {
        "raw": user_input,
        "category": model.category,
        "intent_type": "admin" if is_admin else model.intent_type,
        "summary": model.summary,
        "time_expressions": {} if is_admin else dict(model.time_expressions or {}),
        "risk_level": model.risk_level,
        "needs_rag": bool(needs_rag),
    }
    return {
        "user_intent": intent,
        "context_window": context_window,
        "target_time": target_time,
        "time_prefer": time_prefer,
        "needs_rag": bool(needs_rag),
        "route": "admin" if model.category == "admin" else "context",
        "messages": list(state.get("messages", [])) + [{"role": "user", "content": user_input}],
        "plan_decision": "pending",
        "tool_traces": [],
        "feedback": [],
        "alternatives": [],
        "conflict_rounds": 0,
        "iteration": 0,
        "pending_steps": [],
        "retry_attempt": 0,
        "degraded": False,
        "degraded_question": "",
        "degraded_confirmed": False,
    }


def admin_node(state: dict) -> dict:
    """管理员/测试预置指令：原样记录，完全跳过时间解析与工具 Schema 链路。"""
    bundle = _bundle(state)
    notes = list(bundle.get("admin_notes", [])) + [state["user_input"]]
    bundle["admin_notes"] = notes
    answer = (
        "已记录这条预置信息，未创建任何日程或待办。\n\n"
        f"{state['user_input']}\n\n"
        "该指令按管理员预置分支处理，不进入时间解析与工具参数校验链路；"
        "如需真正写入日程，请直接说“安排……”。"
    )
    return {"context_bundle": bundle, "final_answer": answer}


def context_node(state: dict) -> dict:
    """上下文聚合：只使用程序生成的 context_window，不再接收用户时间原文。"""
    window = state.get("context_window") or {}
    aggregated = aggregate_context(
        window.get("start"),
        window.get("end"),
        max_days=7,
    )
    bundle = _bundle(state)
    bundle.update(
        window=aggregated["window"],
        schedules=aggregated["schedules"],
        todos=aggregated["todos"],
        pending_todos=aggregated["pending_todos"],
        near_deadline_todos=aggregated["near_deadline_todos"],
        stats=aggregated["stats"],
    )
    trace = _trace(
        "aggregate_context",
        aggregated["window"],
        summary=(
            f"日程 {aggregated['stats']['schedule_count']} 条 / 窗口待办 {aggregated['stats']['todo_count']} 条"
            f" / 临近截止 {aggregated['stats']['near_deadline_count']} 条"
        ),
    )
    return {"context_bundle": bundle, "tool_traces": state.get("tool_traces", []) + [trace]}


def rag_node(state: dict) -> dict:
    """按需检索业务文档：只有制度问题或创建/变更规划类才会走到这里。"""
    intent = state["user_intent"]
    query = state["user_input"] if intent.get("category") == "query" else (intent.get("summary") or state["user_input"])
    documents = pipeline.retrieve(query)
    bundle = _bundle(state)
    bundle["documents"] = [
        {"title": doc.title, "content": doc.content, "source": doc.source, "score": doc.score}
        for doc in documents
    ]
    trace = _trace("search_documents", {"query": query}, summary=f"命中 {len(documents)} 条")
    return {
        "context_bundle": bundle,
        "rag_documents": [
            {"title": doc.title, "source": doc.source, "score": doc.score} for doc in documents
        ],
        "tool_traces": state.get("tool_traces", []) + [trace],
    }


def rule_extractor_node(state: dict) -> dict:
    """RAG 后抽取规则，并确定性拆成 hard_constraints / global_baseline_rules。"""
    documents = _bundle(state)["documents"]
    raw_rules: list[dict] = []
    if documents:
        source = "\n\n".join(f"[{doc['title']}] {doc['content'][:900]}" for doc in documents[:4])
        message = get_llm().chat(
            [
                {
                    "role": "system",
                    "content": """你是制度规则抽取器。把文档拆成两套集合，输出严格 JSON：
{"hard_constraints":[{"rule_id":"...","type":"constraint|guideline","category":"meeting|leave|expense|project|general","content":"规则摘要","enforce":"min_lead_hours|work_hours|room_format|min_notice_days|minutes_deadline|approval_flow|expense_window|空字符串","params":{},"valid":true}],
 "global_baseline_rules":[...同结构...]}
hard_constraints 只放本次操作相关的资源互斥、纪要时效、请假提前天数、报销截止、会议室命名、审批链路等。
global_baseline_rules 只放常态化安排，例如“每周一 10:00 周会”“每周五 17:00 前更新进度看板”，它们不计违规、不计分。
enforce 只在能明确量化时填写：会议提前预约 min_lead_hours + {"hours":2}；工作时段 work_hours + {"start":"09:00","end":"18:00"}；会议室命名 room_format；年假提前 min_notice_days + {"days":3}；纪要时效 minutes_deadline + {"hours":24}；报销窗口 expense_window + {"days":30}。
只输出 JSON。""",
                },
                {"role": "user", "content": source},
            ],
            tools=None,
        )
        parsed = parse_llm_json(message["content"], RuleSet)
        if parsed:
            for rule in parsed.hard_constraints + parsed.global_baseline_rules + parsed.rules:
                raw_rules.append(rule.model_dump())

    hard, baseline = split_rules(raw_rules)
    bundle = _bundle(state)
    bundle["hard_constraints"] = hard
    bundle["global_baseline_rules"] = baseline
    bundle["rules"] = hard + baseline
    trace = _trace(
        "rule_extractor",
        {"documents": len(documents)},
        summary=f"硬约束 {len(hard)} 条 / 基线 {len(baseline)} 条",
    )
    return {"context_bundle": bundle, "tool_traces": state.get("tool_traces", []) + [trace]}


def planner_node(state: dict) -> dict:
    """模板优先、LLM 兜底：生成 1~3 个带风险与代价的方案，不执行工具。"""
    intent = state["user_intent"]
    if intent["category"] == "query":
        return {
            "candidate_plans": [], "selected_plan": None, "route": "respond",
            "final_answer": "",
        }

    bundle = _bundle(state)
    hard = _hard_rules(bundle)
    baseline_text = compact_baseline([BusinessRule(**item) for item in bundle["global_baseline_rules"]])
    feedback = state.get("feedback", [])
    extra = ""
    if feedback:
        extra = "\n用户反馈（必须据此重新规划）：" + "；".join(feedback[-3:])

    # 模板库直接给出具体时段与低扰动折中方案，避免 LLM 只给极端方案
    # ug1：把模糊时段偏好注入 intent，供模板按日期+时段区间做 CSP 检索
    template_intent = dict(intent)
    template_intent["time_prefer"] = state.get("time_prefer")
    plans = build_templates(template_intent, bundle, hard)
    if plans:
        return {
            "candidate_plans": plans,
            "route": "evaluate",
            "iteration": state.get("iteration", 0) + 1,
        }

    message = get_llm().chat(
        [
            {
                "role": "system",
                "content": f"""你是办公任务规划器。当前时间 {_now_text()}。
先匹配结构化制度硬约束，再生成 1~3 个可执行方案，输出严格 JSON：
{{"plans":[{{"id":"A","title":"...","description":"...","steps":[{{"tool":"工具名","args":{{}}}}],"risks":[],"cost":"low|medium|high","start_time":"ISO或null","end_time":"ISO或null","needs_confirmation":false,"reschedule_of":null}}]}}

可用工具及参数键（时间字段已按语义拆分，不存在通用 focus_time）：
- create_schedule: title, start_time, end_time, room_prefer, description, attendees
- delete_schedule: id（改期时与 create_schedule 组合，会原子执行）
- create_todo: content, deadline, priority, description, tags, link_schedule_id
- apply_leave: leave_date, leave_type, reason
- check_room_occupancy: check_datetime, room_name
- query_schedules / query_todos: context_window = {{start, end}}（程序生成的区间，不要自己编）
- search_documents: query, top_k
要求：
1. steps 是有序数组，args 必须是 JSON 对象，时间一律用 ISO 8601；不要传 Schema 之外的字段。
2. 会议起止只能放 start_time/end_time；待办截止只能放 deadline；请假日期只能放 leave_date；查会议室只能用 check_datetime。禁止把业务时间塞进 context_window。
3. 下面给出的“结构化目标时间”已经解析完成，直接照抄，不要重新解读用户的时间原文。
3. 优先给出对团队扰动最小的折中方案（如找人代参会 + 会后同步纪要），再给请假/交接、改期等方案。
4. 类别为 change/plan 时必须给出至少 2 个实质不同的方案，并写明风险。
5. 遵守以下结构化**硬约束**（不可违反）：
{compact_rules(hard)}
{baseline_text}
6. {template_hint(intent, bundle, hard)}
7. cost 字段仅供参考，最终代价标签由评估分数自动映射。不要执行工具，只输出方案 JSON。{extra}
只输出 JSON。""",
            },
            {
                "role": "user",
                "content": f"""用户需求：{state['user_input']}
意图：{intent['summary']}（{intent['category']} / {intent['intent_type']}，风险 {intent['risk_level']}）
结构化目标时间（ISO，直接照抄）：{json.dumps(state.get('target_time') or {}, ensure_ascii=False)}
程序推导的查询窗口：{json.dumps(state.get('context_window') or {}, ensure_ascii=False)}
窗口内日程：{_compact(bundle['schedules'])}
相关待办（全部未完成，延后风险用其自身 deadline）：{_compact(bundle['pending_todos'])}
""",
            },
        ],
        tools=None,
    )
    parsed = parse_llm_json(message["content"], PlanSet)
    if parsed is None or not parsed.plans:
        parsed = _fallback_plans(intent, state["user_input"], state.get("target_time"))
    plans = [item.model_dump() for item in parsed.plans]
    return {
        "candidate_plans": plans,
        "route": "evaluate",
        "iteration": state.get("iteration", 0) + 1,
    }


def evaluator_node(state: dict) -> dict:
    """加权打分：只用 hard_constraints 算违规；按可行性→代价→分数排序。"""
    bundle = _bundle(state)
    hard = _hard_rules(bundle)
    evaluated = []
    for plan in state.get("candidate_plans", []):
        evaluation = evaluate_plan(plan, bundle, hard)
        item = {**plan, "evaluation": evaluation}
        if not item.get("cost") or item["cost"] != evaluation["cost"]:
            item["cost"] = evaluation["cost"]  # 分数区间自动映射，覆盖 LLM 硬编码
        evaluated.append(item)

    evaluated.sort(
        key=lambda item: (
            not item["evaluation"]["feasible"],
            _cost_rank(item.get("cost")),
            -item["evaluation"]["score"],
        )
    )
    if not evaluated:
        return {"candidate_plans": [], "selected_plan": None, "route": "respond", "final_answer": "未能生成可执行方案。"}

    best = evaluated[0]
    if not best.get("recommendation") and state.get("alternatives"):
        best["recommendation"] = format_recommendation(state["alternatives"][0], state["alternatives"])

    intent = state["user_intent"]
    route = "execute"
    if not best["evaluation"]["feasible"] and state.get("conflict_rounds", 0) < 1:
        route = "conflict"
    elif (
        intent.get("risk_level") == "high"
        or intent.get("category") in {"change", "plan"}
        or best.get("needs_confirmation")
    ):
        route = "confirm"
    return {"candidate_plans": evaluated, "selected_plan": best, "route": route}


def conflict_resolve_node(state: dict) -> dict:
    """冲突消解：CSP 求 Top3 具体时段 + 可用会议室，生成备选方案。"""
    best = state.get("selected_plan") or {}
    bundle = _bundle(state)
    hard = _hard_rules(bundle)
    intent = state["user_intent"]

    target = state.get("target_time") or {}
    desired = (
        best.get("start_time")
        or target.get("schedule_start")
        or target.get("check_time")
        or intent.get("time_hint")
    )
    window = plan_window(best)
    duration = max(15, int((window.end - window.start).total_seconds() // 60)) if window else 60
    alternatives = suggest_windows(desired, duration, bundle, hard, top_k=3, span_days=5) if desired else []

    if alternatives:
        new_plans = []
        for index, alt in enumerate(alternatives, 1):
            plan = deepcopy(best)
            plan["id"] = f"{best.get('id', 'A')}-R{index}"
            plan["title"] = f"{best.get('title', '方案')} · 备选 {index}"
            plan["start_time"] = alt["start_time"]
            plan["end_time"] = alt["end_time"]
            plan["recommendation"] = format_recommendation(alt, alternatives)
            for step in plan.get("steps", []):
                args = step.get("args", {})
                if "start_time" in args:
                    args["start_time"] = alt["start_time"]
                if "end_time" in args:
                    args["end_time"] = alt["end_time"]
                if alt.get("location") and "location" in args:
                    args["location"] = alt["location"]
            plan["risks"] = list(dict.fromkeys(
                (plan.get("risks") or []) + [f"与原期望相差 {alt['distance_minutes']} 分钟"]
            ))
            new_plans.append(plan)
        feedback = state.get("feedback", []) + [
            f"检测到冲突，已用约束求解器生成 {len(new_plans)} 个可行时段："
            + "；".join(format_alternative(alt) for alt in alternatives)
        ]
        route = "evaluate"
    else:
        feedback = state.get("feedback", []) + ["未找到完全无冲突时段，需用户确认是否继续"]
        new_plans = state.get("candidate_plans", [])
        route = "confirm"

    return {
        "candidate_plans": new_plans,
        "alternatives": alternatives,
        "conflict_rounds": state.get("conflict_rounds", 0) + 1,
        "feedback": feedback,
        "route": route,
    }


def _public_plans(plans: list[dict], selected_id: str | None = None) -> list[dict]:
    """只保留用户可见字段：不暴露原始 trace、权重表与内部调试字段。"""
    result = []
    for plan in plans:
        ev = plan.get("evaluation") or {}
        item = {
            "id": plan.get("id"),
            "title": plan.get("title"),
            "description": plan.get("description", ""),
            "steps": [{"tool": step.get("tool"), "args": step.get("args", {})} for step in plan.get("steps", [])],
            "risks": plan.get("risks", []),
            "cost": plan.get("cost"),
            "start_time": plan.get("start_time"),
            "end_time": plan.get("end_time"),
            "recommendation": plan.get("recommendation"),
            "score": ev.get("score"),
            "feasible": ev.get("feasible"),
            "conflict_count": ev.get("conflict_count"),
            "violation_count": ev.get("violation_count"),
            "recommended": plan.get("id") == selected_id,
        }
        result.append(item)
    return result


def confirm_node(state: dict) -> dict:
    """HITL 断点：高风险写入前暂停，等用户确认后再继续。"""
    plans = _public_plans(state.get("candidate_plans", []), (state.get("selected_plan") or {}).get("id"))
    baseline = [item.get("content") for item in _bundle(state).get("global_baseline_rules", []) if item.get("valid")]
    decision = interrupt(
        {
            "kind": "plan_confirm",
            "question": "请确认执行方案（写入日程/待办前最后一步）",
            "plans": plans,
            "selected_plan_id": (state.get("selected_plan") or {}).get("id"),
            "summary": state["user_intent"].get("summary"),
            "baseline_notes": baseline[:3],
        }
    )
    if isinstance(decision, dict):
        action = str(decision.get("action", "approve")).lower()
        comment = str(decision.get("comment", "")).strip()
        plan_id = decision.get("plan_id")
    else:
        action, comment, plan_id = str(decision).lower(), "", None

    if action in {"reject", "modify"}:
        feedback = state.get("feedback", []) + [comment or "用户否决了当前方案，请重新规划"]
        return {
            "plan_decision": "rejected" if action == "reject" else "modified",
            "feedback": feedback,
            "route": "planner",
        }

    chosen = state.get("selected_plan")
    if plan_id:
        chosen = next((item for item in state.get("candidate_plans", []) if item.get("id") == plan_id), chosen)
    return {"plan_decision": "approved", "selected_plan": chosen, "route": "execute"}


def delete_confirm_node(state: dict) -> dict:
    """Bug6：删除动作的独立 HITL 确认弹窗。确认后只做软删除，不新增任何待办。"""
    intent = state["user_intent"]
    is_schedule = intent.get("intent_type") == "delete_schedule"
    target_label = "日程" if is_schedule else "待办"
    object_name = f"{target_label} ID:{intent.get('target_id')}"
    question = (
        f"是否确认删除{object_name}，删除后仅可在后台找回，前端不再展示？"
    )
    decisions = interrupt(
        {
            "kind": "delete_confirm",
            "question": question,
            "plans": [
                {
                    "id": "A",
                    "title": f"确认删除{object_name}",
                    "description": "软删除：仅标记删除位，不新增任何待办或日程记录；可在后台找回。",
                    "steps": [],
                    "risks": ["删除后前端不再展示", "仅可在后台找回"],
                    "cost": "low",
                    "recommended": True,
                }
            ],
            "selected_plan_id": "A",
            "summary": intent.get("summary"),
            "baseline_notes": [],
        }
    )
    if isinstance(decisions, dict):
        action = str(decisions.get("action", "approve")).lower()
    else:
        action = str(decisions).lower()

    if action in {"reject", "cancel", "取消"}:
        return {
            "plan_decision": "rejected",
            "final_answer": f"已取消删除操作，未对{object_name}做任何改动。",
            "route": "respond",
        }

    tool_name = "delete_schedule" if is_schedule else "delete_todo"
    delete_args = (
        {"id": int(intent["target_id"]), "reason": "用户确认删除"}
    )
    plan = {
        "id": "A",
        "title": f"删除{target_label} ID:{intent['target_id']}",
        "steps": [{"tool": tool_name, "args": delete_args}],
        "needs_confirmation": False,
    }
    return {
        "plan_decision": "approved",
        "selected_plan": plan,
        "route": "execute",
    }


def _atomic_reschedule(plan: dict, traces: list[dict]) -> bool:
    """改期原子方案：新建 + 软删除原日程在同一事务完成。返回是否成功。"""
    original_id = plan.get("reschedule_of")
    create_step = next((step for step in plan.get("steps", []) if step.get("tool") == "create_schedule"), None)
    if not original_id or not create_step:
        return False
    args = create_step.get("args") or {}
    try:
        start = parse_datetime(args["start_time"], "start_time")
        end = parse_datetime(args["end_time"], "end_time") if args.get("end_time") else None
        attendees = args.get("attendees")
        if isinstance(attendees, list):
            attendees = ", ".join(str(item) for item in attendees)
        result = repository.reschedule_schedule(
            int(original_id),
            title=args["title"],
            start_time=start,
            end_time=end,
            location=args.get("location"),
            description=args.get("description"),
            attendees=attendees,
        )
        traces.append(_trace("create_schedule", args, True, f"原子改期：新建日程 #{result['new']['id']}"))
        traces.append(
            _trace("delete_schedule", {"id": int(original_id)}, True, f"原日程 #{original_id} 已软删除，可回溯")
        )
        return True
    except Exception as exc:  # noqa: BLE001
        traces.append(
            _trace("create_schedule", args, False, f"原子改期失败：{type(exc).__name__}: {exc}",
                   structured_error("ATOMIC_RESCHEDULE_FAILED", str(exc), "请核对原日程 ID 与新时间。"))
        )
        return False


def execute_node(state: dict) -> dict:
    """按确认方案串行执行；工具失败进入 retry 子节点，不在这里吞错。"""
    selected = state.get("selected_plan") or {}
    traces = list(state.get("tool_traces", []))
    feedback = list(state.get("feedback", []))
    steps = state.get("pending_steps") or selected.get("steps", [])
    seen: set[str] = set()
    atomic_reschedule_done = False

    for step in steps:
        name = step.get("tool", "")
        args = step.get("args") or {}
        fingerprint = json.dumps({"tool": name, "args": args}, sort_keys=True, ensure_ascii=False, default=str)
        if name == "delete_schedule" and atomic_reschedule_done and str(args.get("id")) == str(selected.get("reschedule_of") or ""):
            # 原子改期已在事务里软删除原日程，跳过独立工具调用，避免重复写入
            continue
        if fingerprint in seen:
            result = structured_error("DUPLICATE_CALL", f"重复调用 {name}，已拦截", "请改用不同参数。")
        elif name == "create_schedule" and selected.get("reschedule_of"):
            if _atomic_reschedule(selected, traces):
                atomic_reschedule_done = True
                continue
            result = traces[-1]["result"]
        else:
            result = call_tool(name, args)

        try:
            payload = json.loads(result)
            ok = bool(payload.get("ok", False))
            outcome_text = str(payload.get("message") or (f"{name} 成功" if ok else f"{name} 失败"))
        except Exception:  # noqa: BLE001
            ok = False
            outcome_text = f"{name} 失败"
        seen.add(fingerprint)
        traces.append(_trace(name, args, ok, outcome_text, result))
        if not ok:
            feedback.append(f"工具 {name} 返回错误：{result[:300]}")
            return {
                "tool_traces": traces,
                "feedback": feedback,
                "pending_steps": steps[steps.index(step):],
                "retry_attempt": 0,
                "route": "retry",
            }

    return {
        "tool_traces": traces,
        "feedback": feedback,
        "pending_steps": [],
        "retry_attempt": 0,
        "route": "respond",
    }


def retry_node(state: dict) -> dict:
    """重试子节点：最多 2 次，自动归一化 ISO 时间、剔除多余 key。"""
    pending = list(state.get("pending_steps", []))
    if not pending:
        return {"route": "respond"}
    step = pending[0]
    name = str(step.get("tool", ""))
    args = step.get("args") or {}
    attempt = int(state.get("retry_attempt", 0)) + 1
    traces = list(state.get("tool_traces", []))
    feedback = list(state.get("feedback", []))

    tool = get_tool(name)
    if tool is None:
        sanitized = dict(args)
    else:
        sanitized = sanitize_args(tool.args_model, args)
    result = call_tool(name, sanitized)
    try:
        ok = json.loads(result).get("ok", False)
    except Exception:  # noqa: BLE001
        ok = False

    if ok:
        traces.append(_trace(name, sanitized, True, f"第 {attempt} 次自动修正后成功", result))
        feedback.append(f"工具 {name} 经自动修正后成功（剔除多余字段 / 归一化时间）")
        return {
            "tool_traces": traces, "feedback": feedback,
            "pending_steps": pending[1:], "retry_attempt": 0,
            "route": "execute" if len(pending) > 1 else "respond",
        }

    traces.append(_trace(name, sanitized, False, f"自动修正重试 {attempt}/2 失败", result))
    if attempt < 2:
        feedback.append(f"工具 {name} 第 {attempt} 次重试仍失败，继续自动修正")
        return {
            "tool_traces": traces, "feedback": feedback,
            "pending_steps": pending, "retry_attempt": attempt, "route": "retry",
        }

    feedback.append(f"工具 {name} 连续 2 次重试失败，已请求用户强确认")
    degraded_question = (
        f"「{name}」暂时失败，本次方案冲突无法完整校验，是否继续执行？"
        if name in {"query_schedules", "query_todos", "search_documents"}
        else f"「{name}」连续 2 次执行失败，是否跳过该步骤继续？"
    )
    return {
        "tool_traces": traces, "feedback": feedback,
        "pending_steps": pending[1:], "retry_attempt": 0,
        "degraded": True, "degraded_question": degraded_question,
        "route": "degraded_confirm",
    }


def degraded_confirm_node(state: dict) -> dict:
    """二次强确认：工具不可用时防止盲写 DB。"""
    question = state.get("degraded_question") or "工具执行失败，是否继续？"
    decision = interrupt(
        {
            "kind": "degraded_confirm",
            "question": question,
            "warning": "继续执行可能在未完整校验冲突的情况下写入数据；建议选择取消并人工核对。",
            "pending_steps": [step.get("tool") for step in state.get("pending_steps", [])],
        }
    )
    if isinstance(decision, dict):
        action = str(decision.get("action", "abort")).lower()
    else:
        action = str(decision).lower()
    confirmed = action in {"continue", "approve", "yes", "继续", "y"}
    feedback = state.get("feedback", []) + [
        "用户确认：工具失败后继续（已知冲突未完整校验）" if confirmed else "用户中止：工具失败后放弃写入"
    ]
    return {
        "degraded_confirmed": True,
        "feedback": feedback,
        "pending_steps": state.get("pending_steps", []) if confirmed else [],
        "route": "execute" if confirmed and state.get("pending_steps") else "respond",
    }


_PRIORITY_LABEL = {"high": "高", "medium": "中等", "low": "低"}


def _dedup_todos(items: list[dict]) -> list[dict]:
    """ug7：按 todo.id 去重，同一待办只保留一条；无 id 时用 (标题, 截止) 兜底。"""
    seen: set[tuple] = set()
    result: list[dict] = []
    for item in items or []:
        key = (
            ("id", item["id"])
            if item.get("id") is not None
            else ("anon", str(item.get("title")), str(item.get("due_time")))
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _delay_guidance(bundle: dict, hard: list[BusinessRule]) -> str:
    """双维度提示：用户业务优先级 + 制度时限，风险交给用户，不直接判定可延后。

    数据源固定为【全局 pending_todos】，不是窗口内 todos：
    延后评估关心的是所有未完成任务的制度时限，与用户查询哪一天无关。
    """
    lines: list[str] = []
    # 数据源固定为全局未完成待办；兼容旧调用方仅传 todos 的情形
    todos = bundle.get("pending_todos") or bundle.get("todos", [])
    # ug7：以 todo.id 为唯一 key 去重，同一个待办只渲染一次【可权衡延后】卡片；
    # 无 id 时用 (标题, 截止时间) 兜底，避免重复数据导致卡片重复渲染。
    seen_keys: set[tuple] = set()
    # 只看日期临近的未完成待办；纪要/报销等受制度硬约束的任务即使中等优先级也要提示
    for todo in todos:
        key = (
            ("id", todo["id"])
            if todo.get("id") is not None
            else ("anon", str(todo.get("title")), str(todo.get("due_time")))
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if not todo.get("due_time") or todo.get("time_status") == "overdue":
            continue
        days_left = todo.get("days_left")
        if days_left is None:
            try:
                hours = (parse_datetime(todo["due_time"], "due_time") - datetime.now()).total_seconds() / 3600
                days_left = int(hours // 24)
            except Exception:  # noqa: BLE001
                continue
        if days_left < 0 or days_left > 7:
            continue
        priority = str(todo.get("priority") or "medium")
        title = str(todo.get("title") or "")
        rule = None
        for item in hard:
            if item.enforce in {"minutes_deadline", "expense_window"} or any(
                word in item.content for word in ("纪要", "报销")
            ):
                if "纪要" in title and (item.enforce == "minutes_deadline" or "纪要" in item.content):
                    rule = item
                    break
                if "报销" in title and (item.enforce == "expense_window" or "报销" in item.content):
                    rule = item
                    break
        if rule is None and priority != "high":
            continue

        due_text = todo.get("due_display") or todo["due_time"][:16]
        priority_label = _PRIORITY_LABEL.get(priority, priority)
        if rule and rule.enforce == "expense_window":
            window_days = int(rule.params.get("days", 30))
            latest = (parse_datetime(todo["due_time"], "due_time") + timedelta(days=window_days)).strftime("%m.%d")
            ceiling = (
                f"制度上限：{rule.content.split('，')[0]}，最晚可延至 {latest} 日（按宽限 {window_days} 天估算）"
            )
            risk = "延后可能影响财务对账，建议评估后决定"
        elif rule and rule.enforce == "minutes_deadline":
            ceiling = (
                f"制度上限：{rule.content.split('，')[0]}，会议纪要需在会后 24 小时内完成，"
                "截止时间为会议结束后 24 小时，不可超出该时限。"
            )
            risk = "纪要延后会影响信息同步与他人跟进，建议按期完成"
        else:
            ceiling = "制度上限：已检索制度中未找到明确宽限窗口，请以主管确认为准"
            risk = "延后可能影响项目排期与他人依赖，建议评估后决定"
        lines.append(
            f"【可权衡延后】{title}（{priority_label}优先｜截止：{due_text}）\n"
            f"业务标记：{priority_label}优先级（建议尽量按期完成）\n"
            f"{ceiling}\n"
            f"⚠️ 风险提示：{risk}"
        )
        if len(lines) >= 2:
            break
    return "\n\n".join(lines)


def _deadline_section(bundle: dict, hard: list[BusinessRule], include_later: bool = False) -> str:
    """确定性地生成「临近截止 / 可权衡延后」段落，避免 LLM 把关联日程或远期待办误标为临近。

    临近截止只来自全局 near_deadline_todos，与用户查询窗口完全解耦；
    每条都标注基于自身 deadline 的真实剩余时长，不引用关联日程时间。

    include_later 默认关闭：「后续到期（供参考，非临近截止）」不是告警信息，
    只在用户明确要看全部/后续待办清单时才展示，避免每条回答都出现。
    """
    near = _dedup_todos(bundle.get("near_deadline_todos", []))
    pending = _dedup_todos(bundle.get("pending_todos", []))
    lines: list[str] = ["### 临近截止任务"]

    def remaining(item: dict) -> str:
        if item.get("time_status") == "overdue":
            return f"已逾期 {abs(item.get('days_left') or 0)} 天"
        if item.get("time_status") == "near":
            hours = item.get("hours_left")
            return f"临近截止（不足 {settings.near_deadline_hours} 小时，剩余 {hours} 小时）" if hours is not None else "临近截止"
        return f"距离现在还有 {item.get('days_left')} 天"

    if near:
        for item in near:
            lines.append(
                f"- {item.get('title')}｜截止：{item.get('due_display')}（{remaining(item)}）"
                f"｜优先级：{_PRIORITY_LABEL.get(str(item.get('priority')), item.get('priority'))}"
                f"｜状态：{item.get('status')}"
            )
    else:
        lines.append(f"- 全部 {len(pending)} 条未完成待办中，暂无 {settings.near_deadline_hours} 小时内临近截止的任务。")

    if include_later:
        later = [
            item for item in pending
            if not item.get("is_near_deadline") and item.get("due_display") != "未设置截止时间"
        ]
        if later:
            lines.append("")
            lines.append("### 后续到期（供参考，非临近截止）")
            for item in later[:5]:
                lines.append(f"- {item.get('title')}｜截止：{item.get('due_display')}（{remaining(item)}）")

    delay_text = _delay_guidance(bundle, hard)
    if delay_text:
        lines.extend(["", delay_text])
    return "\n".join(lines)


# 只有用户主动问到临近截止/延后/待办时间时，才在回答末尾附加待办时间段落；
# 普通建日程、查日程、制度问答等不再出现“临近截止任务/后续到期”。
_DEADLINE_TOPIC_KEYWORDS = (
    "临近", "截止", "到期", "延期", "推迟", "延后", "宽限", "时限",
    "待办", "任务", "提醒", "超时", "逾期", "deadline",
)
_LATER_TODO_KEYWORDS = (
    "全部待办", "所有待办", "后续待办", "后续到期", "还有哪些待办",
    "有哪些待办", "未完成待办", "列一下待办", "待办清单",
)


def _deadline_display(state: dict) -> tuple[bool, bool]:
    """返回 (是否附加待办时间段落, 是否附带「后续到期」参考清单)。"""
    text = str(state.get("user_input") or "")
    intent_type = str((state.get("user_intent") or {}).get("intent_type") or "")
    topic_hit = intent_type == "query_todo" or any(word in text for word in _DEADLINE_TOPIC_KEYWORDS)
    if not topic_hit:
        return False, False
    return True, any(word in text for word in _LATER_TODO_KEYWORDS)


def respond_node(state: dict) -> dict:
    """汇总用户可见事实生成回答；调试信息不进入 prompt 与回答。"""
    bundle = _bundle(state)
    intent = state["user_intent"]
    selected = state.get("selected_plan") or {}
    hard = _hard_rules(bundle)
    documents = bundle["documents"][:3]
    history = [m for m in state.get("messages", []) if m.get("role") in {"user", "assistant"}][-6:]

    # 只把执行结果转写成自然语言，原始 trace / JSON / 文档权重一律不外泄
    outcomes = [
        f"{trace.get('tool')}：{'成功' if trace.get('ok') else '失败'}"
        + (f"（{trace['summary']}）" if trace.get("summary") else "")
        for trace in state.get("tool_traces", [])
        if trace.get("tool") not in {"aggregate_context", "search_documents", "rule_extractor"}
    ]
    public_plan = _public_plans([selected], selected.get("id"))[0] if selected else {}
    baseline_text = compact_baseline([BusinessRule(**item) for item in bundle["global_baseline_rules"]])

    prompt = f"""基于以下事实，用简洁中文回答用户。要求：
1. 查询类直接给数据结论，并写出用户询问的具体日期（如 2026-11-24）；
   创建/变更类先说明执行结果，日程编号与时间必须严格取自“执行结果”，绝对不要编造编号。
2. 若用户已确认方案，说明选择了哪个方案及原因（可行性与相对代价），不要罗列打分权重或内部调试字段。
3. 提到冲突时给出具体时段与会议室；不要说空泛的“下午/下周”。
4. 引用制度要点时标明来源文档；常态化基线只能作为附加提醒，绝不能表述为本次方案违规。
5. 【待办时间硬约束】“某天的待办”只依据“窗口内待办”判断，其 deadline 才可信；
   “全局未完成待办”仅用于关联信息展示，绝不能当成用户问的那一天的待办。
   ❌ 严禁用关联日程的 start_time/end_time 去判断待办归属某一天，或判断待办是否临近截止。
   若窗口内待办为空，必须明确说“暂无截止在该时间段的待办”，不要把全局待办塞进该列表。
6. 不要在回答中自行编写“临近截止任务”“后续到期”“【可权衡延后】”段落——
   这些由系统依据全局 deadline 独立计算；只有用户主动问到临近/截止/延后/待办时才附在末尾，
   普通建日程、查日程、制度问答不要出现，你只需回答用户的主问题。
7. 如工具失败，如实说明并给出建议，绝不编造未执行的操作。
8. 回答控制在 400 字以内，使用 Markdown，不要输出工具名、trace、分数权重、相关度等调试信息。

用户问题：{state['user_input']}
意图：{intent['summary']}（{intent['category']}）
时间窗：{bundle['window']}
日程：{_compact(bundle['schedules'], 8)}
窗口内待办（这是唯一可用于回答“某天待办”的数据）：{_compact(bundle['todos'], 8)}
已选方案：{json.dumps(public_plan, ensure_ascii=False)[:1200]}
执行结果：{json.dumps(outcomes, ensure_ascii=False)}
制度文档要点：{json.dumps([{'title': d['title'], 'content': d['content'][:500]} for d in documents], ensure_ascii=False)[:2200]}
附加提醒（常态化基线，非违规）：{baseline_text}
历史对话：{json.dumps(history, ensure_ascii=False)[:1200]}
"""
    message = get_llm().chat(
        [
            {"role": "system", "content": "你是严谨的办公助理，只依据给定事实回答，不输出任何调试信息。"},
            {"role": "user", "content": prompt},
        ],
        tools=None,
    )
    answer = message["content"] or "（未生成回答）"
    # 临近截止 / 可权衡延后 由系统确定性生成，且只在用户主动问到相关话题时附加；
    # 普通回答不再出现“后续到期（供参考，非临近截止）”这类噪音。
    show_deadline, include_later = _deadline_display(state)
    if show_deadline:
        deadline_section = _deadline_section(bundle, hard, include_later=include_later)
        if deadline_section:
            answer = answer.rstrip() + "\n\n" + deadline_section
    return {"final_answer": answer}
