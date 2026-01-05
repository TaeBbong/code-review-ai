from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class DiffChunk:
    """
    Map 단계에서 처리할 diff 청크 단위.

    Attributes:
        file_path: 파일 경로 (예: "backend/pipelines/base.py")
        content: 해당 파일의 diff 내용
        metadata: 추가 메타데이터 (라인 수, 변경 타입 등)
    """
    file_path: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
