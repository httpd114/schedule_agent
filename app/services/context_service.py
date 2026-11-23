"""上下文聚合查询 Service。

一次查询拉取时间窗内全部日程与待办，替代多轮工具轮询。
窗口（context_window）只接受已解析的 ISO 8601 / datetime，
不再接收用户原始自然语言时间串；窗口由 parse_nl_time 按业务时间自动生成。

⚠️ 待办时间语义硬约束：
- 待办筛选 / 排序 / 临近告警的唯一可信源是 todo.due_time（todo.deadline）。
- 关联日程时间只用于展示，绝不参与任何待办时间过滤或告警计算。
- todos / pending_todos / near_deadline_todos 三个集合语义互不混淆：
    todos             = deadline 严格落在查询窗口内（回答“某天的待办”）
    pending_todos     = 全局未完成（仅供延后提示与关联展示，不能当窗口结果）
    near_deadline_todos = 全局 deadline 临近（与查询窗口完全解耦，独立判断）
"""
from __future__ import annotations

from datetime import datetime, timedelta

from app.models import repository
from app.utils.validators import parse_optional_datetime
from config import settings


def _valid_schedule(item: dict) -> bool:
    """过滤脏日程：空标题、单字符脏数据、无法解析的时间、已取消记录。"""
    if str(item.get("status") or "") == "cancelled":
        return False
    if len(str(item.get("title") or "").strip()) < 2:
        return False
    try:
        start = parse_optional_datetime(item.get("start_time"), "start_time", default_hour=0)
        if start is None:
            return False
        if item.get("end_time"):
            end = parse_optional_datetime(item["end_time"], "end_time", default_hour=23)
            if end is not None and end <= start:
                return False
    except Exception:  # noqa: BLE001
        return False
    return True


def _with_deadline_meta(item: dict, now: datetime) -> dict:
    """给待办附加基于【自身 deadline】的距离信息，供临近判断与展示直接使用。

    这里绝不读取任何关联日程时间；没有 deadline 的待办不参与任何时间告警。
    """
    enriched = dict(item)
    due = parse_optional_datetime(item.get("due_time"), "due_time", default_hour=18)
    if due is None:
        enriched.update(
            due_display="未设置截止时间", hours_left=None, days_left=None,
            time_status="no_deadline", is_near_deadline=False,
        )
        return enriched

    hours_left = (due - now).total_seconds() / 3600
    if hours_left < 0:
        status = "overdue"
    elif hours_left <= settings.near_deadline_hours:
        status = "near"
    else:
        status = "later"
    enriched.update(
        due_display=due.strftime("%Y-%m-%d %H:%M"),
        hours_left=round(hours_left, 1),
        # 按实际剩余小时折算整天：11-23 21:00 → 11-27 10:00 约 85h，显示为 3 天
        days_left=int(hours_left // 24),
        time_status=status,
        is_near_deadline=status in {"near", "overdue"},
    )
    return enriched


def aggregate_context(
    window_start: str | datetime | None = None,
    window_end: str | datetime | None = None,
    keyword: str | None = None,
    limit: int = 100,
    max_days: int = 7,
    now: datetime | None = None,
) -> dict:
    """聚合时间窗内的日程 + 三个语义分离的待办集合。

    window_start/window_end 必须已由 parse_nl_time 解析，
    这里只做区间裁剪与数据过滤，不再解析任何用户原始时间表达。
    """
    current = now or datetime.now()
    start = parse_optional_datetime(window_start, "window_start", default_hour=0)
    end = parse_optional_datetime(window_end, "window_end", default_hour=23)
    if start is None:
        start = current.replace(hour=0, minute=0, second=0, microsecond=0)
    if end is None:
        end = start + timedelta(days=max_days - 1, hours=23, minutes=59)
    if end < start:
        end = start + timedelta(days=1)
    if (end.date() - start.date()).days > max_days:
        end = start + timedelta(days=max_days - 1, hours=23, minutes=59)

    schedules = [
        item for item in repository.list_schedules(
            keyword=keyword, start_from=start, start_to=end, limit=limit
        )
        if _valid_schedule(item)
    ]

    # ① 窗口内待办：仓储层按 Todo.due_time 做区间过滤，关联日程时间完全不参与。
    #    没有 deadline 的待办不会被区间条件命中，自然不属于任何“某天的待办”。
    window_todos = [
        _with_deadline_meta(item, current)
        for item in repository.list_todos(due_after=start, due_before=end, limit=limit)
    ]

    # ② 全局未完成待办：独立集合，只供延后风险提示与关联信息展示，
    #    绝不并入 window_todos，避免“27 号截止”被当成“明天的待办”。
    pending = [
        _with_deadline_meta(item, current)
        for item in repository.list_todos(status="pending", limit=limit)
    ]

    # ③ 临近截止：在【全部未完成待办】上独立判断，不受用户查询窗口影响。
    #    阈值来自配置 NEAR_DEADLINE_HOURS；距离基于各自 deadline 计算。
    near_deadline = [item for item in pending if item["is_near_deadline"]]
    near_deadline.sort(key=lambda item: (item["hours_left"] is None, item["hours_left"] or 0))

    return {
        "window": {"start": start.isoformat(timespec="seconds"), "end": end.isoformat(timespec="seconds")},
        "schedules": schedules,
        "todos": window_todos,
        "pending_todos": pending,
        "near_deadline_todos": near_deadline,
        "stats": {
            "schedule_count": len(schedules),
            "todo_count": len(window_todos),
            "pending_todo_count": len(pending),
            "near_deadline_count": len(near_deadline),
            "high_priority_count": sum(1 for todo in window_todos if todo.get("priority") == "high"),
        },
    }
