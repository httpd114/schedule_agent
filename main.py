#!/usr/bin/env python3
"""日程任务自动编排 Agent —— 命令行入口（支持 HITL 方案确认）。

用法：
    python main.py --init              # 可选：写入演示数据（默认不执行，保持空库）
    python main.py --demo              # 运行全部示例问题（自动批准方案）
    python main.py "明天下午3点安排项目复盘会"   # 单次提问
    python main.py --debug "..."       # 开发者模式：显示工具 trace 等调试信息
    python main.py                     # 交互模式
"""
from __future__ import annotations

import argparse
import sys

from app.graph.workflow import run_agent
from app.models import repository
from app.models.database import init_db
from app.tools.registry import tool_names

EXAMPLES = [
    "查一下我明天有哪些日程，还有哪些待办",
    "帮我创建一个后天上午10点的产品评审日程，并加一条会后发纪要的待办",
    "公司报销的审批额度和时限是怎样的？",
    "我无法参加下周一10点的周会，帮我推演几种方案",
    "记个待办：周五前提交差旅报销，高优先级",
]


def _print_banner() -> None:
    print("=" * 62)
    print("  日程任务自动编排 Agent  |  Plan → Evaluate → HITL → Execute")
    print(f"  已注册工具: {', '.join(tool_names())}")
    print("  输入 exit / quit 退出；输入 examples 查看示例")
    print("=" * 62)


def _show_plans(result: dict) -> None:
    print("\n--- 待确认方案 ---")
    for plan in result.get("plans", []):
        flag = " [推荐]" if plan.get("recommended") else ""
        print(f"  [{plan.get('id')}] {plan.get('title')}{flag} · 代价 {plan.get('cost')}")
        if plan.get("recommendation"):
            print(f"      {plan['recommendation']}")
        if plan.get("risks"):
            print("      风险：" + "；".join(plan["risks"]))
    for note in result.get("baseline_notes", [])[:3]:
        print(f"  （附加提醒，非违规）{note}")


def _ask_decision(result: dict, auto_approve: bool = False) -> dict:
    if result.get("kind") == "degraded_confirm":
        if auto_approve:
            return {"action": "abort"}
        print(f"\n{result.get('question')}")
        print(f"提示：{result.get('warning', '')}")
        choice = input("输入 c 确认继续（校验不完整），其它键取消：>>> ").strip().lower()
        return {"action": "continue" if choice in {"c", "continue", "y", "继续"} else "abort"}

    if auto_approve:
        return {"action": "approve", "plan_id": result.get("selected_plan_id")}
    ids = [plan.get("id") for plan in result.get("plans", [])]
    print("\n输入方案编号批准，n 否决，m 修改后重规划，回车批准推荐方案：")
    choice = input(">>> ").strip().lower()
    if choice in {"n", "no", "否决"}:
        return {"action": "reject", "comment": "用户否决，请重新推演方案"}
    if choice in {"m", "modify", "修改"}:
        comment = input("请输入修改意见：").strip() or "用户要求调整方案"
        return {"action": "modify", "comment": comment}
    if choice in ids:
        return {"action": "approve", "plan_id": choice}
    return {"action": "approve", "plan_id": result.get("selected_plan_id")}


def _show(result: dict, debug: bool = False) -> None:
    chosen = result.get("selected_plan") or {}
    if chosen:
        print(f"\n已执行方案：{chosen.get('title')}")
        if chosen.get("recommendation"):
            print(f"  {chosen['recommendation']}")
    if debug and result.get("debug"):
        info = result["debug"]
        print("\n--- 调试信息（仅 --debug 可见） ---")
        for trace in info.get("tool_calls", []):
            print(f"  {trace.get('tool')} ok={trace.get('ok')} {trace.get('summary', '')}")
        print(f"  文档命中: {[d.get('title') for d in info.get('rag_documents', [])]}")
        print(f"  反馈: {info.get('feedback', [])}")
    print("\n--- 最终回答 ---")
    print(result.get("answer", ""))
    print()


def ask(question: str, history: list[dict] | None = None, auto_approve: bool = False,
        debug: bool = False) -> dict:
    """发起请求；遇到 HITL 断点时展示方案并按用户决策继续。"""
    result = run_agent(question, history=history, include_debug=debug)
    while result.get("status") == "awaiting_confirmation":
        if result.get("kind") == "degraded_confirm":
            decision = _ask_decision(result, auto_approve=auto_approve)
        else:
            _show_plans(result)
            decision = _ask_decision(result, auto_approve=auto_approve)
        result = run_agent(
            "", thread_id=result["thread_id"], decision=decision, include_debug=debug
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="日程任务自动编排 Agent")
    parser.add_argument("--init", action="store_true", help="初始化演示数据")
    parser.add_argument("--demo", action="store_true", help="运行内置示例问题")
    parser.add_argument("--debug", action="store_true", help="显示工具轨迹等调试信息")
    parser.add_argument("input", nargs="?", help="单次提问")
    args = parser.parse_args()

    init_db()
    if args.init:
        from scripts.seed import seed

        cleaned = repository.purge_invalid_schedules()
        print(f"[cleanup] 已将 {cleaned} 条脏日程标记为取消。")
        seed()
        return 0

    if args.demo:
        for question in EXAMPLES:
            print(f"\n>>> {question}")
            try:
                _show(ask(question, auto_approve=True, debug=args.debug), debug=args.debug)
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 0

    if args.input:
        _show(ask(args.input, debug=args.debug), debug=args.debug)
        return 0

    _print_banner()
    history: list[dict] = []
    while True:
        try:
            question = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "bye"}:
            print("再见。")
            break
        if question.lower() == "examples":
            for idx, example in enumerate(EXAMPLES, 1):
                print(f"  {idx}. {example}")
            continue
        try:
            result = ask(question, history=history, debug=args.debug)
            _show(result, debug=args.debug)
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": result.get("answer", "")})
            history = history[-10:]
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
