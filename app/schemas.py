import time
from typing import List, Optional, Union, Dict, Any, Literal
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = 512
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    speculative_k: Optional[int] = None
    use_speculative: Optional[bool] = True


class CompletionRequest(BaseModel):
    model: str = "default"
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    use_speculative: Optional[bool] = True
    speculative_k: Optional[int] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    speculative_accepted_tokens: Optional[int] = 0
    speculative_drafted_tokens: Optional[int] = 0
    acceptance_rate: Optional[float] = 0.0


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "content_filter"]] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


class ChatCompletionChunkDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[Literal["stop", "length", "content_filter"]] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChunkChoice]
    usage: Optional[UsageInfo] = None


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: Optional[str] = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "speculative-proxy"
    permission: List[Dict[str, Any]] = []
    root: Optional[str] = None
    parent: Optional[str] = None


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelCard]


class ErrorObject(BaseModel):
    message: str
    type: str = "invalid_request_error"
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorObject
