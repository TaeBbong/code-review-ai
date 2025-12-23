import re

def extract_json_text(raw: str) -> str:
    raw = raw.strip()
    # ```json ... ``` 제거
    m = re.search(r"```json\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    # 일반 ``` ... ``` 제거
    m = re.search(r"```\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw
