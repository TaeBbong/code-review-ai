# HumanEvalFix Dataset

HumanEvalFix 데이터셋을 코드 리뷰 평가용으로 변환한 데이터셋입니다.

## 원본 데이터셋

- **이름**: [bigcode/humanevalpack](https://huggingface.co/datasets/bigcode/humanevalpack)
- **출처**: OctoPack 프로젝트
- **라이선스**: MIT

## 데이터셋 특징

- **언어**: Python
- **샘플 수**: 164개
- **카테고리**: correctness (버그 탐지)
- **버그 유형**:
  - missing logic: 누락된 로직
  - excess logic: 불필요한 로직
  - operator misuse: 연산자 오용
  - variable misuse: 변수 오용
  - value misuse: 값 오용
  - function misuse: 함수 오용

## 변환 방법

```bash
# datasets 패키지 설치
pip install datasets

# 변환 스크립트 실행
uv run python -m backend.evaluation.datasets.humanevalfix.convert
```

## 변환 로직

1. HuggingFace에서 `bigcode/humanevalpack` Python 데이터셋 로드
2. 각 샘플의 `canonical_solution` (정답)과 `buggy_solution` (버그 코드) 비교
3. `difflib.unified_diff`로 diff 생성
4. `bug_type`을 severity와 keywords로 매핑
5. 우리 평가 스키마에 맞게 YAML 출력

## 평가 실행

```bash
# 변환 후 평가 실행
uv run python -m backend.evaluation.cli run-local humanevalfix_python g0-baseline
```

## 주의사항

- diff는 **정상 코드 → 버그 코드** 방향입니다 (버그를 도입하는 변경)
- LLM이 "이 변경은 버그를 도입한다"고 지적해야 정답
- 모든 샘플은 `correctness` 카테고리입니다
