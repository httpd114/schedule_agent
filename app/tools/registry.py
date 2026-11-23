"""工具注册表：新增工具只需继承 BaseTool 并加 @register_tool 装饰器。"""
from __future__ import annotations

from app.tools.base import BaseTool

_REGISTRY: dict[str, BaseTool] = {}


def register_tool(cls: type[BaseTool]) -> type[BaseTool]:
    """类装饰器，实例化后按 name 注册。"""
    instance = cls()
    if not instance.name:
        raise ValueError(f"工具 {cls.__name__} 未定义 name")
    if instance.name in _REGISTRY:
        raise ValueError(f"工具 {instance.name} 重复注册")
    _REGISTRY[instance.name] = instance
    return cls


def get_tool(name: str) -> BaseTool | None:
    return _REGISTRY.get(name)


def call_tool(name: str, arguments: dict) -> str:
    tool = _REGISTRY.get(name)
    if tool is None:
        return __import__("json").dumps(
            {
                "ok": False,
                "error_code": "UNKNOWN_TOOL",
                "message": f"不存在名为 {name} 的工具",
                "hint": f"可用工具: {sorted(_REGISTRY)}",
            },
            ensure_ascii=False,
        )
    return tool(**arguments)


def openai_tools() -> list[dict]:
    return [tool.to_openai_schema() for tool in _REGISTRY.values()]


def tool_names() -> list[str]:
    return sorted(_REGISTRY)
