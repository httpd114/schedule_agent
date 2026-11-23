"""LLM 结构化输出解析：剥离代码围栏与前后缀，再用 Pydantic 校验。"""
from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def parse_llm_json(text: str, model: type[T], fallback: T | None = None) -> T | None:
    if not text:
        return fallback
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```JSON", "").replace("```", "")
    starts = [pos for pos in (cleaned.find("{"), cleaned.find("[")) if pos >= 0]
    ends = [pos for pos in (cleaned.rfind("}"), cleaned.rfind("]")) if pos >= 0]
    if not starts or not ends:
        return fallback
    fragment = cleaned[min(starts) : max(ends) + 1]
    try:
        return model.model_validate(json.loads(fragment))
    except (json.JSONDecodeError, ValidationError):
        return fallback
