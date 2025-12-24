from __future__ import annotations

from pydantic import ValidationError
from langchain_core.runnables import Runnable

from backend.core.helpers import extract_json_text
from backend.schemas.review import ReviewResult


async def validate_or_repair(
    *,
    raw_text: str,
    repair_chain: Runnable,
    bad_max_chars: int = 4000,
):
    """
    Returns:
      (parsed_model, repair_used, raw_json, fixed_json_or_none)
    """
    raw_json: str = extract_json_text(raw_text)

    try:
        parsed = ReviewResult.model_validate_json(raw_json)
        return parsed, False, raw_json, None
    except ValidationError:
        fixed_msg = await repair_chain.ainvoke({"bad": raw_json[:bad_max_chars]})
        fixed_json = extract_json_text(fixed_msg.content)
        parsed = ReviewResult.model_validate_json(fixed_json)
        return parsed, True, raw_json, fixed_json
