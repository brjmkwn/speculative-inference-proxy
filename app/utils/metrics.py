import time
from typing import Optional, List
import torch
from prometheus_client import Counter, Histogram, Gauge


REQUEST_COUNT = Counter(
    "proxy_requests_total",
    "Total requests processed",
    ["endpoint", "model", "mode", "status"]
)

REQUEST_LATENCY = Histogram(
    "proxy_request_duration_seconds",
    "Request duration in seconds",
    ["endpoint", "mode"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
)

TTFT_HISTOGRAM = Histogram(
    "proxy_time_to_first_token_seconds",
    "Time to first token in seconds",
    ["mode"],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.5, 1.0]
)

ITL_HISTOGRAM = Histogram(
    "proxy_inter_token_latency_seconds",
    "Inter-token latency in seconds",
    ["mode"],
    buckets=[0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.5]
)

TOKENS_GENERATED_TOTAL = Counter(
    "proxy_tokens_generated_total",
    "Tokens generated",
    ["type"]
)

SPECULATIVE_ACCEPTANCE_RATE = Gauge(
    "proxy_speculative_acceptance_rate",
    "Acceptance rate of draft tokens"
)

SPECULATION_CYCLES_TOTAL = Counter(
    "proxy_speculation_cycles_total",
    "Total speculation cycles executed"
)

GPU_VRAM_USAGE = Gauge(
    "proxy_gpu_vram_usage_bytes",
    "GPU memory allocated in bytes",
    ["device_id"]
)


def update_gpu_metrics():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i)
            GPU_VRAM_USAGE.labels(device_id=str(i)).set(allocated)


class RequestTelemetry:
    def __init__(self, mode: str = "speculative"):
        self.mode = mode
        self.start_time: float = time.perf_counter()
        self.first_token_time: Optional[float] = None
        self.last_token_time: Optional[float] = None
        
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.draft_tokens_proposed: int = 0
        self.draft_tokens_accepted: int = 0
        self.cycles: int = 0
        self.inter_token_latencies: List[float] = []

    def record_first_token(self):
        now = time.perf_counter()
        if self.first_token_time is None:
            self.first_token_time = now
            ttft = now - self.start_time
            TTFT_HISTOGRAM.labels(mode=self.mode).observe(ttft)
            self.last_token_time = now

    def record_token_emitted(self, count: int = 1):
        now = time.perf_counter()
        if self.first_token_time is None:
            self.record_first_token()
            self.completion_tokens += count
            return

        if self.last_token_time is not None and count > 0:
            itl = (now - self.last_token_time) / count
            self.inter_token_latencies.append(itl)
            ITL_HISTOGRAM.labels(mode=self.mode).observe(itl)

        self.completion_tokens += count
        self.last_token_time = now

    def record_speculation_cycle(self, proposed: int, accepted: int):
        self.cycles += 1
        self.draft_tokens_proposed += proposed
        self.draft_tokens_accepted += accepted
        
        SPECULATION_CYCLES_TOTAL.inc()
        TOKENS_GENERATED_TOTAL.labels(type="speculative_drafted").inc(proposed)
        TOKENS_GENERATED_TOTAL.labels(type="speculative_accepted").inc(accepted)
        
        if self.draft_tokens_proposed > 0:
            rate = self.draft_tokens_accepted / self.draft_tokens_proposed
            SPECULATIVE_ACCEPTANCE_RATE.set(rate)

    @property
    def total_duration(self) -> float:
        return time.perf_counter() - self.start_time

    @property
    def ttft(self) -> float:
        if self.first_token_time is None:
            return self.total_duration
        return self.first_token_time - self.start_time

    @property
    def tokens_per_second(self) -> float:
        duration = self.total_duration
        if duration <= 0:
            return 0.0
        return self.completion_tokens / duration

    @property
    def acceptance_rate(self) -> float:
        if self.draft_tokens_proposed == 0:
            return 0.0
        return self.draft_tokens_accepted / self.draft_tokens_proposed
