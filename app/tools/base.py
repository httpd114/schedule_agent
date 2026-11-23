"""插件化工具基类：Pydantic 校验 + 结构化错误协议。"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from app.tools.schemas import ToolArgs, function_parameters
from app.utils.validators import ValidationError


def success_payload(**payload) -> str:
    """工具成功返回：始终 JSON，便于模型稳定解析。"""
    return json.dumps({"ok": True, **payload}, ensure_ascii=False)


def structured_error(code: str, message: str, hint: str = "", **extra: Any) -> str:
    """工具失败返回：给模型足够的上下文以自愈修正。"""
    return json.dumps(
        {"ok": False, "error_code": code, "message": message, "hint": hint, **extra},
        ensure_ascii=False,
    )


class ToolExecutionError(Exception):
    def __init__(self, code: str, message: str, hint: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint


class BaseTool(ABC):
    name: str = ""
    description: str = ""
    args_model: type[ToolArgs] = ToolArgs

    @property
    def parameters(self) -> dict:
        return function_parameters(self.args_model)

    def __call__(self, **kwargs) -> str:
        """统一校验与异常兜底：任何异常都转成结构化 JSON，绝不把 traceback 抛给模型。"""
        try:
            validated = self.args_model.model_validate(kwargs)
        except PydanticValidationError as exc:
            errors = [
                {
                    "loc": ".".join(str(part) for part in item["loc"]),
                    "msg": item["msg"],
                    "type": item["type"],
                }
                for item in exc.errors()
            ]
            return structured_error(
                "VALIDATION_ERROR",
                f"工具 {self.name} 的参数校验失败（validation_error）",
                "请依据 errors 字段逐项修正参数；时间建议使用 ISO 8601，枚举值请严格匹配。",
                error_type="validation_error",
                errors=errors,
            )

        try:
            return self.execute(**validated.model_dump())
        except ValidationError as exc:
            return structured_error("INVALID_ARGUMENT", exc.message, exc.hint)
        except ToolExecutionError as exc:
            return structured_error(exc.code, exc.message, exc.hint)
        except Exception as exc:  # noqa: BLE001 - 兜底是刻意设计
            return structured_error(
                "TOOL_EXCEPTION",
                f"{type(exc).__name__}: {exc}",
                "请检查工具参数是否正确；必要时改用其它工具或缩小查询范围。",
            )

    @abstractmethod
    def execute(self, **kwargs) -> str:
        ...

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
