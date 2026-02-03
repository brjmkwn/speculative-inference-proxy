import time
from fastapi import APIRouter, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.config import settings
from app.core.engine import engine
from app.schemas import ModelList, ModelCard
from app.utils.metrics import update_gpu_metrics

router = APIRouter(tags=["telemetry"])


@router.get("/health")
async def health():
    update_gpu_metrics()
    return {
        "status": "healthy" if engine.is_ready else "initializing",
        "ready": engine.is_ready,
        "device": engine.device,
        "dtype": str(engine.dtype),
        "target_model": settings.TARGET_MODEL_ID,
        "draft_model": settings.DRAFT_MODEL_ID,
        "lookahead_k": engine.lookahead_k,
        "mock_engine": engine.use_mock,
        "timestamp": int(time.time())
    }


@router.get("/metrics")
async def metrics():
    update_gpu_metrics()
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/v1/models", response_model=ModelList)
async def list_models():
    return ModelList(
        data=[
            ModelCard(id="default", owned_by="speculative-proxy", root=settings.TARGET_MODEL_ID),
            ModelCard(id=settings.TARGET_MODEL_ID, owned_by="speculative-proxy"),
            ModelCard(id=settings.DRAFT_MODEL_ID, owned_by="speculative-proxy", parent=settings.TARGET_MODEL_ID),
        ]
    )
