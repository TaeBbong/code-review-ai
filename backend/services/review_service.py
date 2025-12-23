from __future__ import annotations
import logging

from pydantic import ValidationError
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from backend.core.settings import settings
from backend.core.helpers import extract_json_text
from backend.core.llm.base import AdapterChatModel
from backend.core.llm.provider import get_llm_adapter
from backend.schemas.review import ReviewRequest, ReviewResult


logger = logging.getLogger()


SYSTEM_PROMPT = "You are a strict code review bot. Return ONLY JSON. No markdown."
REPAIR_SYSTEM = "You output only JSON. No extra text."

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """Review the following git diff and output JSON that matches the schema.

{format_instructions}

variant_id: {variant_id}

git diff:
{diff}
"""),
])

REPAIR_PROMPT = """Fix your previous output to match the required JSON schema.

{format_instructions}

Rules:
- Return ONLY JSON. No markdown. No commentary.
- Do NOT add extra keys.
- Ensure correct types (e.g., summary is an object; test_suggestions is a list of objects).

Bad output:
{bad}
"""


class ReviewService:
    async def review(self, req: ReviewRequest) -> ReviewResult:
        llm = AdapterChatModel(get_llm_adapter(), model_name=settings.ollama_model)
        parser = PydanticOutputParser(pydantic_object=ReviewResult)
        format_instructions = parser.get_format_instructions()

        chain = PROMPT.partial(format_instructions=format_instructions) | llm

        msg = await chain.ainvoke({"variant_id": req.variant_id, "diff": req.diff})
        raw = extract_json_text(msg.content)

        try:
            return ReviewResult.model_validate_json(raw)
        except ValidationError as e:
            logger.warning("Validation failed (first pass): %s", e)

        repair_chain = ChatPromptTemplate.from_messages([
            ("system", REPAIR_SYSTEM),
            ("human", REPAIR_PROMPT),
        ]).partial(format_instructions=format_instructions) | llm

        fixed_msg = await repair_chain.ainvoke({"bad": raw[:4000]})
        fixed = extract_json_text(fixed_msg.content)

        return ReviewResult.model_validate_json(fixed)