from __future__ import annotations

import httpx
from langchain_core.runnables import Runnable


async def invoke_chain(chain: Runnable, payload: dict):
    try:
        return await chain.ainvoke(payload)
    except (httpx.ConnectError, httpx.ReadTimeout, ConnectionError) as e:
        raise RuntimeError("LLM backend is unavailable") from e
