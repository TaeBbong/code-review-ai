from __future__ import annotations
import logging
import httpx
from datetime import datetime

from pydantic import ValidationError
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from backend.core.context import run_id_var
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
        run_id = run_id_var.get()
        llm = AdapterChatModel(get_llm_adapter(), model_name=settings.ollama_model)
        parser = PydanticOutputParser(pydantic_object=ReviewResult)
        format_instructions = parser.get_format_instructions()

        chain = PROMPT.partial(format_instructions=format_instructions) | llm

        msg = await chain.ainvoke({"variant_id": req.variant_id, "diff": req.diff})
        raw = extract_json_text(msg.content)
        logger.info("LLM run_id=%s variant=%s raw_len=%d", run_id, req.variant_id, len(raw))

        repair_used = False

        try:
            msg = await chain.ainvoke({"variant_id": req.variant_id, "diff": req.diff})
        except (httpx.ConnectError, httpx.ReadTimeout, ConnectionError) as e:
            logger.exception("LLM_UNAVAILABLE run_id=%s err=%s", run_id, str(e))
            raise RuntimeError("LLM backend is unavailable") from e

        raw = extract_json_text(msg.content)
        logger.info(
            "LLM_OUTPUT run_id=%s variant=%s raw_len=%d raw_head=%r",
            run_id, req.variant_id, len(raw), raw[:200]
        )

        try:
            result = ReviewResult.model_validate_json(raw)
        except ValidationError as e:
            repair_used = True
            logger.warning("PARSE_FAIL run_id=%s variant=%s err=%s", run_id, req.variant_id, str(e))

            repair_chain = ChatPromptTemplate.from_messages([
                ("system", REPAIR_SYSTEM),
                ("human", REPAIR_PROMPT),
            ]).partial(format_instructions=format_instructions) | llm

            fixed_msg = await repair_chain.ainvoke({"bad": raw[:4000]})
            fixed = extract_json_text(fixed_msg.content)

            logger.info(
                "LLM_REPAIR run_id=%s fixed_len=%d fixed_head=%r",
                run_id, len(fixed), fixed[:200]
            )
            result = ReviewResult.model_validate_json(fixed)

        result.meta.variant_id = req.variant_id
        result.meta.model = settings.ollama_model
        result.meta.diff_target = "raw"
        result.meta.generated_at = datetime.utcnow().isoformat()

        logger.info(
            "DONE run_id=%s variant=%s repair_used=%s issues=%d tests=%d questions=%d",
            run_id, req.variant_id, repair_used,
            len(result.issues), len(result.test_suggestions), len(result.questions_to_author)
        )

        return result