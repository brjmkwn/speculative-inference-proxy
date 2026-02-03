import uuid
import time
import json
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException, Depends, Security
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import settings
from app.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    UsageInfo,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    CompletionRequest,
    CompletionResponse,
    CompletionChoice
)
from app.core.engine import engine
from app.utils.logger import logger
from app.utils.metrics import RequestTelemetry, REQUEST_COUNT, REQUEST_LATENCY

router = APIRouter(prefix="/v1", tags=["completions"])
security = HTTPBearer(auto_error=False)


def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    if settings.API_KEY is not None:
        if not credentials or credentials.credentials != settings.API_KEY:
            raise HTTPException(
                status_code=401,
                detail={"error": {"message": "Invalid API key", "type": "invalid_request_error"}}
            )
    return credentials


async def sse_chat_generator(
    request: ChatCompletionRequest,
    prompt: str,
    telemetry: RequestTelemetry,
    req_id: str
) -> AsyncGenerator[str, None]:
    created_time = int(time.time())
    model_id = request.model or settings.TARGET_MODEL_ID
    
    initial_chunk = ChatCompletionChunk(
        id=req_id,
        created=created_time,
        model=model_id,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(role="assistant", content=""),
                finish_reason=None
            )
        ]
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"

    stop_seqs = [request.stop] if isinstance(request.stop, str) else (request.stop or [])
    
    if request.use_speculative:
        gen = engine.generate_speculative_stream(
            prompt=prompt,
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature if request.temperature is not None else 0.7,
            top_p=request.top_p if request.top_p is not None else 1.0,
            k=request.speculative_k,
            stop_strings=stop_seqs,
            telemetry=telemetry
        )
    else:
        gen = engine.generate_baseline_stream(
            prompt=prompt,
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature if request.temperature is not None else 0.7,
            top_p=request.top_p if request.top_p is not None else 1.0,
            stop_strings=stop_seqs,
            telemetry=telemetry
        )

    try:
        async for chunk_text, _ in gen:
            if not chunk_text:
                continue
            chunk = ChatCompletionChunk(
                id=req_id,
                created=created_time,
                model=model_id,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(content=chunk_text),
                        finish_reason=None
                    )
                ]
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        final_chunk = ChatCompletionChunk(
            id=req_id,
            created=created_time,
            model=model_id,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(),
                    finish_reason="stop"
                )
            ],
            usage=UsageInfo(
                prompt_tokens=telemetry.prompt_tokens,
                completion_tokens=telemetry.completion_tokens,
                total_tokens=telemetry.prompt_tokens + telemetry.completion_tokens,
                speculative_accepted_tokens=telemetry.draft_tokens_accepted,
                speculative_drafted_tokens=telemetry.draft_tokens_proposed,
                acceptance_rate=round(telemetry.acceptance_rate, 4)
            )
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error("Error in SSE stream: %s", e, exc_info=True)
        err_chunk = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {json.dumps(err_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        mode = "speculative" if request.use_speculative else "vanilla"
        REQUEST_COUNT.labels(endpoint="/v1/chat/completions", model=model_id, mode=mode, status="200").inc()
        REQUEST_LATENCY.labels(endpoint="/v1/chat/completions", mode=mode).observe(telemetry.total_duration)


@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    _auth: Any = Depends(verify_api_key)
):
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())
    model_id = request.model or settings.TARGET_MODEL_ID
    mode = "speculative" if request.use_speculative else "vanilla"

    messages_payload = [{"role": m.role, "content": m.content} for m in request.messages]
    formatted_prompt = engine.format_chat_prompt(messages_payload)

    telemetry = RequestTelemetry(mode=mode)

    if request.stream:
        return StreamingResponse(
            sse_chat_generator(request, formatted_prompt, telemetry, req_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
                "X-Accel-Buffering": "no"
            }
        )

    try:
        generated_text, telemetry = await engine.generate(
            prompt=formatted_prompt,
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature if request.temperature is not None else 0.7,
            top_p=request.top_p if request.top_p is not None else 1.0,
            use_speculative=bool(request.use_speculative),
            k=request.speculative_k,
            stop=request.stop
        )

        REQUEST_COUNT.labels(endpoint="/v1/chat/completions", model=model_id, mode=mode, status="200").inc()
        REQUEST_LATENCY.labels(endpoint="/v1/chat/completions", mode=mode).observe(telemetry.total_duration)

        return ChatCompletionResponse(
            id=req_id,
            created=created_time,
            model=model_id,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=generated_text),
                    finish_reason="stop"
                )
            ],
            usage=UsageInfo(
                prompt_tokens=telemetry.prompt_tokens,
                completion_tokens=telemetry.completion_tokens,
                total_tokens=telemetry.prompt_tokens + telemetry.completion_tokens,
                speculative_accepted_tokens=telemetry.draft_tokens_accepted,
                speculative_drafted_tokens=telemetry.draft_tokens_proposed,
                acceptance_rate=round(telemetry.acceptance_rate, 4)
            )
        )

    except Exception as e:
        logger.error("Chat completion error: %s", e, exc_info=True)
        REQUEST_COUNT.labels(endpoint="/v1/chat/completions", model=model_id, mode=mode, status="500").inc()
        raise HTTPException(status_code=500, detail={"error": {"message": str(e), "type": "server_error"}})


@router.post("/completions")
async def raw_completions(
    request: CompletionRequest,
    _auth: Any = Depends(verify_api_key)
):
    req_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
    mode = "speculative" if request.use_speculative else "vanilla"

    generated_text, telemetry = await engine.generate(
        prompt=prompt,
        max_tokens=request.max_tokens or 256,
        temperature=request.temperature if request.temperature is not None else 0.7,
        top_p=request.top_p if request.top_p is not None else 1.0,
        use_speculative=bool(request.use_speculative),
        k=request.speculative_k,
        stop=request.stop
    )

    return CompletionResponse(
        id=req_id,
        model=request.model or settings.TARGET_MODEL_ID,
        choices=[CompletionChoice(index=0, text=generated_text, finish_reason="stop")],
        usage=UsageInfo(
            prompt_tokens=telemetry.prompt_tokens,
            completion_tokens=telemetry.completion_tokens,
            total_tokens=telemetry.prompt_tokens + telemetry.completion_tokens,
            speculative_accepted_tokens=telemetry.draft_tokens_accepted,
            speculative_drafted_tokens=telemetry.draft_tokens_proposed,
            acceptance_rate=round(telemetry.acceptance_rate, 4)
        )
    )
