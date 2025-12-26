from fastapi import APIRouter, HTTPException

from backend.domain.schemas.review import ReviewRequest, ReviewResult
from backend.domain.services.review_service import ReviewService


router = APIRouter()
service = ReviewService()


@router.get("/health")
async def health():
    return {"ok": True}


@router.get("/schema/review")
async def review_schema():
    return {
        "request": ReviewRequest.model_json_schema(),
        "response": ReviewResult.model_json_schema(),
    }


@router.post("/review", response_model=ReviewResult)
async def review(req: ReviewRequest):
    try:
        return await service.review(req)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    