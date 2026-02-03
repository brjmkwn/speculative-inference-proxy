from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.utils.logger import logger
from app.core.engine import engine
from app.routers.chat import router as chat_router
from app.routers.telemetry import router as telemetry_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Starting proxy service (target=%s, draft=%s, K=%d, device=%s)",
        settings.TARGET_MODEL_ID,
        settings.DRAFT_MODEL_ID,
        settings.LOOKAHEAD_K,
        settings.resolved_device()
    )
    engine.load_models()
    yield
    logger.info("Stopping proxy service")


app = FastAPI(
    title="Speculative Decoding Inference Proxy",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": "server_error",
                "param": None,
                "code": "internal_error"
            }
        }
    )


app.include_router(chat_router)
app.include_router(telemetry_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower()
    )
