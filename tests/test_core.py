"""离线测试：时间校验 / 工具 Schema / 规则拆分 / 加权打分 / 模板规划 / 重试降级。"""
from __future__ import annotations

import inspect
import json
from datetime import datetime, timedelta

import pytest

from app.models import repository
from app.models.database import init_db
from app.tools.registry import call_tool, openai_tools, tool_names
from app.tools.schemas import CreateTodoArgs, QuerySchedulesArgs, sanitize_args
from app.utils.validators import ValidationError, parse_datetime


@pytest.fixture(scope="module", autouse=True)
def _cleanup_test_rows():
    """测试会在真实 SQLite 写入日程；结束后清除明确的测试标题，避免污染用户数据。"""
    yield
    from app.models.database import SessionLocal
    from app.models.entities import Schedule

    with SessionLocal() as session:
        session.query(Schedule).filter(
            Schedule.title.like("测试日程-%") | (Schedule.title == "原子改期测试日程")
        ).delete(synchronize_session=False)
        session.commit()


def test_parse_datetime_formats():
    now = datetime(2026, 11, 21, 9, 0)
    assert parse_datetime("2026-11-22T14:30:00", now=now).hour == 14
    assert parse_datetime("2026/11/22 14:30", now=now).minute == 30
    assert parse_datetime("2026年11月22日 下午3点", now=now).hour == 15
    assert parse_datetime("明天 9:30", now=now).date() == datetime(2026, 11, 22).date()
    assert parse_datetime("后天下午3点", now=now).hour == 15


def test_parse_datetime_rejects_garbage():
    with pytest.raises(ValidationError) as exc:
        parse_datetime("下周三左右")
    assert exc.value.hint


def test_registry_has_eight_tools_with_semantic_time_schemas():
    # ug1/Bug6 新增 reschedule_schedule 与 delete_todo，共 10 个工具
    assert sorted(tool_names()) == [
        "apply_leave", "check_room_occupancy", "create_schedule", "create_todo",
        "delete_schedule", "delete_todo", "query_schedules", "query_todos",
        "reschedule_schedule", "search_documents",
    ]
    schemas = {item["function"]["name"]: item["function"]["parameters"] for item in openai_tools()}
    assert all(item.get("additionalProperties") is False for item in schemas.values())

    create = schemas["create_schedule"]["properties"]
    assert {"title", "start_time", "end_time", "room_prefer"} <= set(create)
    assert "location" not in create and "focus_time" not in create

    todo = schemas["create_todo"]["properties"]
    assert {"content", "deadline", "link_schedule_id"} <= set(todo)
    assert "title" not in todo and "due_time" not in todo

    query = schemas["query_schedules"]["properties"]
    assert "context_window" in query and "focus_time" not in query
    window = query["context_window"]
    assert "$ref" not in window  # Function Calling schema 必须内联
    assert "start" in window.get("properties", {}) and "end" in window.get("properties", {})

    assert {"leave_date", "leave_type"} <= set(schemas["apply_leave"]["properties"])
    assert {"check_datetime", "room_name"} <= set(schemas["check_room_occupancy"]["properties"])


def test_tool_returns_structured_validation_error():
    result = json.loads(call_tool("create_schedule", {"title": "测试", "start_time": "不是时间"}))
    assert result["ok"] is False
    assert result["error_type"] == "validation_error"
    assert result["errors"] and result["errors"][0]["loc"] == "start_time"


def test_string_tags_are_coerced_to_list():
    model = CreateTodoArgs.model_validate({"title": "测试", "tags": "财务,差旅"})
    assert model.tags == ["财务", "差旅"]


def test_sanitize_args_drops_extra_keys_and_normalizes_time():
    cleaned = sanitize_args(
        QuerySchedulesArgs,
        {"date": "明天", "limit": "5", "unknown_key": 123},
    )
    assert "unknown_key" not in cleaned
    assert cleaned["limit"] == 5
    assert "T" in cleaned["date"]


def test_repository_roundtrip_and_soft_delete():
    init_db()
    title = f"测试日程-{datetime.now().timestamp()}"
    created = repository.create_schedule(title=title, start_time=datetime(2027, 1, 1, 10))
    assert created["id"] > 0
    deleted = repository.delete_schedule(created["id"], reason="测试软删除")
    assert deleted["status"] == "cancelled"
    assert all(item["id"] != created["id"] for item in repository.list_schedules(keyword=title))


def test_repository_rejects_dirty_title():
    with pytest.raises(ValueError):
        repository.create_schedule(title="x", start_time=datetime(2027, 1, 1, 10))


def test_graph_builds_and_debug_is_opt_in():
    from app.graph.workflow import build_workflow, run_agent

    assert build_workflow() is not None
    assert inspect.signature(run_agent).parameters["include_debug"].default is False


def test_duplicate_call_is_intercepted():
    from app.graph.nodes import execute_node

    step = {"tool": "query_todos", "args": {"status": "all"}}
    result = execute_node({"selected_plan": {"steps": [step, step]}, "tool_traces": [], "feedback": []})
    assert any("DUPLICATE_CALL" in trace["result"] for trace in result["tool_traces"])
    assert sum(1 for trace in result["tool_traces"] if trace["ok"]) == 1
    assert result["route"] == "retry"


def test_context_aggregation_filters_dirty_schedules():
    from app.services.context_service import aggregate_context

    context = aggregate_context("2026-11-22", "2026-11-30")
    assert "window" in context
    assert all(len(item["title"].strip()) >= 2 for item in context["schedules"])
    assert all(item.get("status") != "cancelled" for item in context["schedules"])
    assert "schedule_count" in context["stats"]


def test_context_window_is_generated_by_parser_not_raw_text():
    from app.services.context_service import aggregate_context
    from app.utils.nl_time import parse_nl_time

    window, target = parse_nl_time(
        {"schedule_start": "11月26日14:00–15:30"},
        "create_schedule", now=datetime(2026, 11, 23, 9, 0),
    ).to_state()
    assert target["schedule_start"] == "2026-11-26T14:00:00"
    assert target["schedule_end"] == "2026-11-26T15:30:00"
    assert window["start"].startswith("2026-11-25")
    context = aggregate_context(window["start"], window["end"])
    start = datetime.fromisoformat(context["window"]["start"])
    end = datetime.fromisoformat(context["window"]["end"])
    assert (end.date() - start.date()).days <= 2


def test_constraint_solver_detects_conflict_and_suggests_alternatives():
    from app.planning.constraint_solver import TimeWindow, detect_conflicts, suggest_windows

    context = {
        "schedules": [{
            "id": 1, "title": "周会", "start_time": "2026-11-23T10:00:00",
            "end_time": "2026-11-23T11:00:00", "location": "12F-A301", "attendees": "张伟",
        }],
        "todos": [],
    }
    conflicts = detect_conflicts(
        TimeWindow(datetime(2026, 11, 23, 10, 30), datetime(2026, 11, 23, 11, 30)),
        context["schedules"], location="12F-A301", attendees="张伟",
    )
    assert conflicts and "time_overlap" in conflicts[0]["types"]
    assert "room_busy" in conflicts[0]["types"]
    assert "attendee_busy" in conflicts[0]["types"]

    alternatives = suggest_windows("2026-11-23T10:30:00", 60, context, [], top_k=3, span_days=1)
    assert alternatives
    for alt in alternatives:
        assert alt.get("location")
        start = datetime.fromisoformat(alt["start_time"])
        end = datetime.fromisoformat(alt["end_time"])
        assert not (start < datetime(2026, 11, 23, 11) and datetime(2026, 11, 23, 10) < end)


def test_rules_split_into_hard_and_baseline():
    from app.planning.rules import split_rules

    hard, baseline = split_rules([
        {"rule_id": "weekly_meeting", "content": "每周一 10:00 召开项目周会", "type": "guideline"},
        {"rule_id": "friday_board", "content": "每周五 17:00 前更新进度看板", "type": "guideline"},
        {"rule_id": "room_format", "content": "会议室命名格式为楼层-编号", "enforce": "room_format"},
        {"rule_id": "leave_notice", "content": "年假需提前 3 个工作日申请", "enforce": "min_notice_days"},
    ])
    assert [item["rule_id"] for item in baseline] == ["weekly_meeting", "friday_board"]
    assert {item["rule_id"] for item in hard} == {"room_format", "leave_notice"}
    assert all(item["scope"] == "hard" for item in hard)
    assert all(item["scope"] == "baseline" for item in baseline)


def test_evaluator_ignores_baseline_and_uses_weighted_score():
    from app.planning.constraint_solver import evaluate_plan
    from app.planning.rules import BusinessRule

    context = {
        "schedules": [{
            "id": 1, "title": "周会", "start_time": "2026-11-26T10:00:00",
            "end_time": "2026-11-26T11:00:00", "location": "12F-A301",
        }],
        "todos": [],
    }
    plan = {
        "title": "周四技术评审", "start_time": "2026-11-26T14:00:00",
        "end_time": "2026-11-26T16:00:00",
        "steps": [{"tool": "create_schedule", "args": {"title": "周四技术评审"}}],
    }
    baseline = [BusinessRule(rule_id="weekly_meeting", content="每周一 10:00 周会", scope="baseline")]
    now = datetime(2026, 11, 21, 9)
    result = evaluate_plan(plan, context, baseline, now=now)
    assert result["violation_count"] == 0
    assert result["score"] == 100  # 周四 14 点不与周一 10 点周会冲突

    hard = [BusinessRule(rule_id="room_format", content="会议室命名格式", enforce="room_format",
                         scope="hard", params={"pattern": r"^\d+[Ff]-"})]
    plan["steps"][0]["args"]["location"] = "不合规会议室"
    result = evaluate_plan(plan, context, hard, now=now)
    assert result["violation_count"] == 1
    assert result["violation_score"] == 3  # 命名格式权重 3
    assert result["score"] == 97
    assert result["cost"] == "low"  # >=80 自动映射，而非 LLM 硬编码


def test_weighted_cost_mapping():
    from app.planning.constraint_solver import _cost_from_score

    assert _cost_from_score(100) == "low"
    assert _cost_from_score(70) == "medium"
    assert _cost_from_score(30) == "high"


def test_meeting_absence_templates_prefer_lightweight_plan():
    from app.planning.rules import BusinessRule
    from app.planning.templates import build_templates

    bundle = {
        "schedules": [{
            "id": 7, "title": "Q4 项目周会", "start_time": "2026-11-23T10:00:00",
            "end_time": "2026-11-23T11:00:00", "location": "12F-A301",
        }],
        "todos": [],
    }
    intent = {
        "category": "plan", "summary": "无法参加周一 10 点的周会",
        "raw": "我无法参加下周一10点的周会，帮我推演几种方案",
        "time_hint": "下周一10点", "risk_level": "high",
    }
    plans = build_templates(
        intent, bundle,
        [BusinessRule(rule_id="room_format", content="会议室命名格式", scope="hard")],
    )
    assert plans and len(plans) == 3
    assert [plan["cost"] for plan in plans] == ["low", "medium", "high"]
    assert "代参会" in plans[0]["title"]
    assert any("改期" in plan["title"] for plan in plans)
    assert all(plan.get("recommendation") for plan in plans)


def test_delay_guidance_keeps_two_dimensions():
    from app.graph.nodes import _delay_guidance
    from app.planning.rules import BusinessRule

    bundle = {
        "todos": [{
            "title": "提交差旅报销", "priority": "high",
            "due_time": (datetime.now() + timedelta(days=1)).isoformat(timespec="seconds"),
        }],
        "schedules": [], "documents": [], "hard_constraints": [], "global_baseline_rules": [],
    }
    hard = [BusinessRule(
        rule_id="expense_window", content="费用发生后应在 30 个自然日内提交报销申请",
        enforce="expense_window", params={"days": 30}, scope="hard",
    )]
    text = _delay_guidance(bundle, hard)
    assert "【可权衡延后】" in text
    assert "业务标记：高优先级" in text
    assert "制度上限：" in text
    assert "⚠️ 风险提示：" in text


def test_rag_route_skips_queries_without_policy_keywords():
    from app.graph.workflow import _rag_route

    assert _rag_route({"needs_rag": False}) == "rule_extractor"
    assert _rag_route({"needs_rag": True}) == "rag"


def test_public_plans_hide_debug_fields():
    from app.graph.nodes import _public_plans

    item = {
        "id": "A", "title": "方案", "steps": [], "cost": "low",
        "evaluation": {"score": 95, "weights": {"secret": 1}},
        "tool_traces": [{"tool": "query_todos"}],
        "rag_documents": [{"title": "内部文档", "score": 0.9}],
    }
    out = _public_plans([item], "A")[0]
    assert out["recommended"] is True
    assert out["score"] == 95
    assert "weights" not in out
    assert "tool_traces" not in out
    assert "rag_documents" not in out


def test_retry_node_escalates_to_strong_confirmation_after_two_failures():
    from app.graph.nodes import retry_node

    step = {"tool": "unknown_tool", "args": {"date": "明天", "extra": 1}}
    result = retry_node({
        "pending_steps": [step], "retry_attempt": 1,
        "tool_traces": [], "feedback": [],
    })
    assert result["degraded"] is True
    assert result["route"] == "degraded_confirm"
    assert "无法完整校验" in result["degraded_question"] or "失败" in result["degraded_question"]


def test_atomic_reschedule_does_not_delete_twice():
    from app.graph.nodes import execute_node

    init_db()
    original = repository.create_schedule(
        title="原子改期测试日程", start_time=datetime(2027, 2, 1, 10),
        end_time=datetime(2027, 2, 1, 11),
    )
    new_start = datetime(2027, 2, 1, 14)
    plan = {
        "reschedule_of": original["id"],
        "steps": [
            {
                "tool": "create_schedule",
                "args": {
                    "title": "原子改期测试日程",
                    "start_time": new_start.isoformat(timespec="seconds"),
                    "end_time": (new_start + timedelta(hours=1)).isoformat(timespec="seconds"),
                },
            },
            {"tool": "delete_schedule", "args": {"id": original["id"]}},
        ],
    }
    result = execute_node({"selected_plan": plan, "tool_traces": [], "feedback": []})
    deletes = [trace for trace in result["tool_traces"] if trace["tool"] == "delete_schedule"]
    assert len(deletes) == 1 and deletes[0]["ok"] is True
    assert result["route"] == "respond"
    assert repository.get_schedule(original["id"])["status"] == "cancelled"


def test_parse_nl_time_business_examples_are_structured():
    from app.utils.nl_time import parse_nl_time

    now = datetime(2026, 11, 23, 19, 0)

    _, meeting = parse_nl_time(
        {"schedule_start": "下周四下午14:00–15:30"}, "create_schedule", now=now
    ).to_state()
    assert meeting["schedule_start"] == "2026-12-03T14:00:00"
    assert meeting["schedule_end"] == "2026-12-03T15:30:00"
    assert meeting["todo_deadline"] is None

    _, todo = parse_nl_time({"todo_deadline": "截止到11月26日"}, "create_todo", now=now).to_state()
    assert todo["todo_deadline"] == "2026-11-26T18:00:00"
    assert todo["schedule_start"] is None

    _, leave = parse_nl_time({"leave_date": "下周一到周三"}, "apply_leave", now=now).to_state()
    assert leave["leave_date"] == "2026-11-30T09:00:00"
    assert leave["leave_end"] == "2026-12-02T18:00:00"

    window, check = parse_nl_time({"check_time": "11.26 14:00"}, "check_room", now=now).to_state()
    assert check["check_time"] == "2026-11-26T14:00:00"
    assert window["start"] == "2026-11-26T00:00:00"
    assert window["end"] == "2026-11-26T23:59:59"


def test_parse_nl_time_query_never_fills_business_time():
    from app.utils.nl_time import parse_nl_time

    window, target = parse_nl_time(
        {"context_start": "12月2号"}, "query_schedule", now=datetime(2026, 11, 23, 9, 0)
    ).to_state()
    assert window["start"].startswith("2026-12-02")
    assert not any(target.values())


def test_admin_expression_is_never_parsed_into_time_fields():
    from app.utils.nl_time import parse_nl_time

    window, target = parse_nl_time(
        "周一 10:00 固定周会、11.26 14:00 15F-B205 已被占用", "admin",
        now=datetime(2026, 11, 23, 9, 0),
    ).to_state()
    assert not any(window.values())
    assert not any(target.values())


def test_state_declares_separated_time_fields():
    from app.graph.state import AgentState

    annotations = AgentState.__annotations__
    assert "context_window" in annotations and "target_time" in annotations
    assert "focus_time" not in annotations


def test_intent_node_routes_time_by_semantics(monkeypatch):
    import json as _json

    from app.graph import nodes

    class FakeLLM:
        def __init__(self, payload):
            self.payload = payload

        def chat(self, messages, tools=None):
            return {"content": _json.dumps(self.payload, ensure_ascii=False)}

    cases = [
        (
            {
                "category": "create", "intent_type": "create_todo",
                "summary": "记个待办", "time_expressions": {"todo_deadline": "截止到11月26日"},
                "risk_level": "low", "needs_rag": False,
            },
            "todo_deadline",
        ),
        (
            {
                "category": "admin", "intent_type": "admin",
                "summary": "预置信息", "time_expressions": {"check_time": "11.26 14:00"},
                "risk_level": "low", "needs_rag": False,
            },
            None,
        ),
    ]
    for payload, expected_key in cases:
        monkeypatch.setattr(nodes, "get_llm", lambda p=payload: FakeLLM(p))
        result = nodes.intent_node({"user_input": "测试输入", "messages": []})
        target = result["target_time"]
        filled = [key for key, value in target.items() if value]
        if expected_key is None:
            assert result["route"] == "admin"
            assert not filled
            assert result["user_intent"]["time_expressions"] == {}
        else:
            assert filled == [expected_key]


def test_check_room_occupancy_tool_and_todo_new_field_names():
    from app.tools.schemas import CreateTodoArgs

    model = CreateTodoArgs.model_validate(
        {"content": "会后同步纪要", "deadline": "11月26日", "link_schedule_id": 1}
    )
    dumped = model.model_dump()
    assert dumped["content"] == "会后同步纪要"
    assert dumped["deadline"].endswith("T18:00:00")
    assert dumped["link_schedule_id"] == 1

    result = json.loads(call_tool(
        "check_room_occupancy",
        {"check_datetime": "2026-11-26T14:00:00", "room_name": "15F-B205"},
    ))
    assert result["ok"] is True
    assert isinstance(result["occupied"], bool)


# ---------- 待办时间语义：窗口筛选 vs 临近截止，唯一可信源 = todo.deadline ----------

@pytest.fixture
def _todo_time_fixture():
    """插入带关联日程语义的待办，结束后按标题清理，避免污染空库。"""
    from datetime import datetime as dt

    from app.models.database import SessionLocal
    from app.models.entities import Todo

    titles = ["测试-关联日程衍生纪要待办", "测试-临近截止待办", "测试-两小时后截止待办"]
    created = []
    with SessionLocal() as session:
        session.add(Todo(
            title=titles[0], due_time=dt(2026, 11, 27, 10, 0), priority="medium", status="pending",
            # 模拟“关联日程在 11-24，但自身截止在 11-27”：description 含关联信息
            description="关联日程 #999（2026-11-24 09:00-10:00）",
        ))
        # 固定基准 now = 11-23 21:00；两条临近待办都在窗口(11-24)之外，
        # 用于验证临近判断与查询窗口完全解耦。
        session.add(Todo(title=titles[1], due_time=dt(2026, 11, 23, 22, 0), priority="high", status="pending"))
        session.add(Todo(title=titles[2], due_time=dt(2026, 11, 23, 23, 0), priority="medium", status="pending"))
        session.commit()
        created = [r.id for r in session.query(Todo).filter(Todo.title.in_(titles)).all()]
    yield created
    with SessionLocal() as session:
        session.query(Todo).filter(Todo.title.in_(titles)).delete(synchronize_session=False)
        session.commit()


def test_window_todos_filter_only_by_deadline_not_linked_schedule(_todo_time_fixture):
    """用户问 11-24 待办：关联日程在 11-24 但 deadline 在 11-27 的待办不得混入。"""
    from app.services.context_service import aggregate_context

    ctx = aggregate_context(
        "2026-11-24T00:00:00", "2026-11-24T23:59:59",
        now=datetime(2026, 11, 23, 21, 0),
    )
    titles = [t["title"] for t in ctx["todos"]]
    assert "测试-关联日程衍生纪要待办" not in titles
    assert all("11-24" in str(t.get("due_time")) or "2026-11-24T" in str(t.get("due_time")) for t in ctx["todos"])


def test_near_deadline_is_global_and_independent_of_window(_todo_time_fixture):
    """临近截止从全部未完成待办独立判断，即使查询窗口是 11-24 也能发现 6 小时后截止的任务。"""
    from app.services.context_service import aggregate_context

    ctx = aggregate_context(
        "2026-11-24T00:00:00", "2026-11-24T23:59:59",
        now=datetime(2026, 11, 23, 21, 0),
    )
    near_titles = [t["title"] for t in ctx["near_deadline_todos"]]
    assert "测试-临近截止待办" in near_titles
    assert "测试-关联日程衍生纪要待办" not in near_titles
    assert ctx["stats"]["near_deadline_count"] == len(ctx["near_deadline_todos"])
    assert "测试-临近截止待办" not in [t["title"] for t in ctx["todos"]]
    assert "测试-两小时后截止待办" not in [t["title"] for t in ctx["todos"]]


def test_near_deadline_threshold_comes_from_config(_todo_time_fixture, monkeypatch):
    """阈值由配置 NEAR_DEADLINE_HOURS 决定，24h 外即使在同一天也不算临近。"""
    from app.services import context_service

    monkeypatch.setattr(context_service.settings, "near_deadline_hours", 1)
    ctx = context_service.aggregate_context(
        "2026-11-24T00:00:00", "2026-11-24T23:59:59", now=datetime(2026, 11, 23, 21, 0)
    )
    near_titles = [t["title"] for t in ctx["near_deadline_todos"]]
    assert "测试-临近截止待办" in near_titles   # 距 now 1h，阈值 1h 内仍算临近
    assert "测试-两小时后截止待办" not in near_titles  # 距 now 2h，超出阈值


def test_deadline_section_never_labels_far_todo_as_near(_todo_time_fixture):
    """确定性段落：27 号截止的待办只能出现在“后续到期”，不能标成临近截止。"""
    from app.graph.nodes import _deadline_section
    from app.planning.rules import BusinessRule
    from app.services.context_service import aggregate_context

    ctx = aggregate_context(
        "2026-11-24T00:00:00", "2026-11-24T23:59:59",
        now=datetime(2026, 11, 23, 21, 0),
    )
    text = _deadline_section(ctx, [BusinessRule(rule_id="r", content="x")], include_later=True)
    near_part, later_part = text.split("### 后续到期", 1)
    assert "测试-关联日程衍生纪要待办" not in near_part
    assert "测试-关联日程衍生纪要待办" in later_part
    assert "距离现在还有 3 天" in later_part

    # 默认不带“后续到期”参考清单，避免每条回答都出现
    default_text = _deadline_section(ctx, [BusinessRule(rule_id="r", content="x")])
    assert "后续到期" not in default_text


def test_deadline_section_is_shown_on_demand():
    """待办时间段落只在用户主动问到临近/截止/待办时附加，且不默认带“后续到期”。"""
    from app.graph.nodes import _deadline_display

    show, later = _deadline_display({"user_input": "帮我安排下周四的评审会", "user_intent": {"intent_type": "create_schedule"}})
    assert show is False and later is False

    show, later = _deadline_display({"user_input": "查一下明天有什么会议", "user_intent": {"intent_type": "query_schedule"}})
    assert show is False and later is False

    show, later = _deadline_display({"user_input": "有哪些临近截止的待办", "user_intent": {"intent_type": "query_todo"}})
    assert show is True and later is False

    show, later = _deadline_display({"user_input": "列一下全部待办", "user_intent": {"intent_type": "query_todo"}})
    assert show is True and later is True


def test_auto_minutes_todo_has_deadline_end_plus_24h():
    """纪要待办自动生成时必须写入 deadline = 会议 end_time + 24h。"""
    from app.planning.templates import _time_window_plans
    from app.planning.rules import BusinessRule

    intent = {
        "category": "create", "intent_type": "create_schedule", "summary": "技术评审",
        "raw": "2026-11-26 14:00 技术评审", "target_time": {"schedule_start": "2026-11-26T14:00:00"},
    }
    bundle = {"schedules": [], "pending_todos": [], "todos": []}
    plans = _time_window_plans(intent, bundle, [BusinessRule(rule_id="r", content="x")])
    assert plans
    minutes = next(
        step for step in plans[0]["steps"] if step.get("tool") == "create_todo"
    )
    end = datetime.fromisoformat(plans[0]["end_time"])
    due = datetime.fromisoformat(minutes["args"]["deadline"])
    assert due - end == timedelta(hours=24)


def test_query_window_covers_full_day_including_early_deadlines():
    """查询窗口必须是完整自然日 00:00-23:59，不能漏掉早上截止的待办。"""
    from app.utils.nl_time import parse_nl_time

    window, target = parse_nl_time(
        {"context_start": "明天"}, "query_todo", now=datetime(2026, 11, 23, 21, 0)
    ).to_state()
    assert window["start"] == "2026-11-24T00:00:00"
    assert window["end"] == "2026-11-24T23:59:59"
    assert not any(target.values())


def test_intent_node_scans_relative_time_when_llm_omits_it(monkeypatch):
    """LLM 漏提“明天”时，确定性兜底仍需生成查询窗口。"""
    import json as _json
    from app.graph import nodes

    class FakeLLM:
        def chat(self, messages, tools=None):
            return {"content": _json.dumps({
                "category": "query", "intent_type": "query_todo", "summary": "查明天待办",
                "time_expressions": {}, "risk_level": "low", "needs_rag": False,
            }, ensure_ascii=False)}

    monkeypatch.setattr(nodes, "get_llm", lambda: FakeLLM())
    result = nodes.intent_node({"user_input": "看看我明天有哪些待办", "messages": []})
    assert result["context_window"]["start"].startswith("2026-11-24")
    assert result["user_intent"]["time_expressions"] == {"context_start": "明天"}


# ---------- ug1 / Bug6 / ug7 验收用例 ----------

def test_ug1_fuzzy_period_returns_prefer_not_validation_error():
    """ug1：「11月27日下午」应产出 prefer 结构，不再抛 ValidationError。"""
    from app.utils.nl_time import parse_nl_time

    parsed = parse_nl_time(
        {"schedule_start": "11月27日下午"}, "reschedule", now=datetime(2026, 11, 23, 21, 0)
    )
    assert parsed.period_prefer is not None
    assert parsed.period_prefer["type"] == "prefer"
    assert parsed.period_prefer["date"] == "2026-11-27"
    assert parsed.period_prefer["period"] == "afternoon"
    # 模糊时段不写精确时间，避免下游 Pydantic 时间校验报错
    assert parsed.target_time["schedule_start"] is None


def test_ug1_period_bounds_mapping():
    """ug1：时段映射 morning 08-12 / afternoon 13-18 / evening 18-21。"""
    from app.utils.time_periods import PERIOD_BOUNDS

    assert PERIOD_BOUNDS["morning"] == (8, 12)
    assert PERIOD_BOUNDS["afternoon"] == (13, 18)
    assert PERIOD_BOUNDS["evening"] == (18, 21)


def test_ug1_reschedule_tool_registered_with_relaxed_time_args():
    """ug1：独立改期工具已注册，target_date/period_prefer 不做严格时间校验。"""
    from app.tools.registry import tool_names
    from app.tools.schemas import RescheduleScheduleArgs

    assert "reschedule_schedule" in tool_names()
    args = RescheduleScheduleArgs.model_validate({
        "schedule_id": 1, "target_date": "2026-11-27", "period_prefer": "afternoon",
        "room_prefer": "12F-A301",
    })
    assert args.target_date == "2026-11-27"
    assert args.period_prefer == "afternoon"


def test_bug6_parse_delete_command_with_remark_text():
    """Bug6：带备注长文本也能抓到 ID，并正确识别目标类型。"""
    from app.graph.nodes import parse_delete_command
    from app.planning.plan_models import ActionType

    res = parse_delete_command("删除整理技术文档 ID:2")
    assert res["action"] == ActionType.DELETE
    assert res["target_id"] == 2
    assert res["target_type"] == "todo"

    res2 = parse_delete_command("取消明天的项目评审会议 ID：5")
    assert res2["target_id"] == 5
    assert res2["target_type"] == "schedule"

    # 无显式 ID 不得误判
    assert parse_delete_command("删除整理技术文档")["action"] is None
    # 日期数字不得被当成 ID
    assert parse_delete_command("删除11月27日的会议")["action"] is None


def test_bug6_delete_intent_shortcuts_to_delete_confirm(monkeypatch):
    """Bug6：删除语句在 Intent 节点短路到 delete_confirm，绝不进 create_todo。"""
    import json as _json
    from app.graph import nodes

    class FakeLLM:
        def chat(self, messages, tools=None):
            # 即便 LLM 误判成 create_todo，确定性识别也必须覆盖它
            return {"content": _json.dumps({
                "category": "create", "intent_type": "create_todo", "summary": "创建删除待办",
                "time_expressions": {}, "risk_level": "low", "needs_rag": False,
            }, ensure_ascii=False)}

    monkeypatch.setattr(nodes, "get_llm", lambda: FakeLLM())
    result = nodes.intent_node({"user_input": "删除整理技术文档 ID:2", "messages": []})
    assert result["route"] == "delete"
    assert result["user_intent"]["category"] == "delete"
    assert result["user_intent"]["intent_type"] == "delete_todo"
    assert result["user_intent"]["target_id"] == 2
    assert result["target_time"] == {}  # 不碰 TARGET_KEYS


def test_bug6_delete_todo_is_soft_delete_and_filtered():
    """Bug6：delete_todo 只置 is_deleted=True，不新增待办；查询默认过滤。"""
    from datetime import datetime as dt
    from app.models.database import SessionLocal
    from app.models.entities import Todo
    from app.models import repository

    with SessionLocal() as s:
        todo = Todo(title="测试-删除验证待办", due_time=dt(2026, 12, 1, 9, 0), status="pending")
        s.add(todo)
        s.commit()
        tid = todo.id
    record = repository.delete_todo(tid, reason="验收测试")
    assert record["is_deleted"] is True
    with SessionLocal() as s:
        titles = [t.title for t in repository.list_todos()]
        assert "测试-删除验证待办" not in titles
        # 确认没有新增“删除 xxx”待办
        assert not any("删除验证待办" in t and t != "测试-删除验证待办" for t in titles)
        s.query(Todo).filter(Todo.id == tid).delete(synchronize_session=False)
        s.commit()


def test_ug7_delay_card_dedup_by_todo_id():
    """ug7：同一 todo.id 重复出现时，延后卡片只渲染一次。"""
    from app.graph.nodes import _dedup_todos

    items = [
        {"id": 1, "title": "A", "due_time": "2026-11-25T10:00:00"},
        {"id": 1, "title": "A", "due_time": "2026-11-25T10:00:00"},
        {"id": 2, "title": "B", "due_time": "2026-11-26T10:00:00"},
    ]
    deduped = _dedup_todos(items)
    assert len(deduped) == 2
    assert [x["id"] for x in deduped] == [1, 2]
