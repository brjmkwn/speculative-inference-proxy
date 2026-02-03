from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import torch


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_KEY: Optional[str] = None
    LOG_LEVEL: str = "INFO"

    TARGET_MODEL_ID: str = "Qwen/Qwen2.5-1.5B-Instruct"
    DRAFT_MODEL_ID: str = "Qwen/Qwen2.5-0.5B-Instruct"
    LOOKAHEAD_K: int = Field(default=4, ge=1, le=16)

    DEVICE: str = "auto"
    DTYPE: str = "auto"
    MAX_MODEL_LEN: int = 4096
    USE_MOCK_ENGINE: bool = False

    def resolved_device(self) -> str:
        if self.DEVICE != "auto":
            return self.DEVICE
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def resolved_dtype(self) -> torch.dtype:
        device = self.resolved_device()
        if self.DTYPE == "bfloat16":
            return torch.bfloat16
        if self.DTYPE == "float16":
            return torch.float16
        if self.DTYPE == "float32":
            return torch.float32
        
        if device == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        elif device == "cuda":
            return torch.float16
        return torch.float32


settings = Settings()
