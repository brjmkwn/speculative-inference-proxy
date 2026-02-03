import pytest
import torch
from fastapi.testclient import TestClient

from app.config import settings
settings.USE_MOCK_ENGINE = True

from app.main import app
from app.core.sampler import speculative_rejection_sample
from app.core.cache import TokenBuffer
from app.core.engine import engine


@pytest.fixture(scope="module", autouse=True)
def init_engine():
    engine.use_mock = True
    engine.load_models()


def test_greedy_sampler_all_accepted():
    k = 4
    vocab_size = 100
    draft_tokens = [10, 20, 30, 40]
    
    draft_logits = torch.zeros(k, vocab_size)
    target_logits = torch.zeros(k + 1, vocab_size)
    for i, tok in enumerate(draft_tokens):
        target_logits[i, tok] = 10.0
    target_logits[k, 50] = 10.0
    
    accepted, num_accepted_draft, num_proposed = speculative_rejection_sample(
        draft_tokens=draft_tokens,
        draft_logits=draft_logits,
        target_logits=target_logits,
        temperature=0.0
    )
    
    assert num_proposed == 4
    assert num_accepted_draft == 4
    assert accepted == [10, 20, 30, 40, 50]


def test_greedy_sampler_partial_rejection():
    k = 4
    vocab_size = 100
    draft_tokens = [10, 20, 30, 40]
    
    draft_logits = torch.zeros(k, vocab_size)
    target_logits = torch.zeros(k + 1, vocab_size)
    
    target_logits[0, 10] = 10.0
    target_logits[1, 20] = 10.0
    target_logits[2, 99] = 10.0  # Mismatch: target is 99, draft was 30
    target_logits[3, 40] = 10.0
    
    accepted, num_accepted_draft, num_proposed = speculative_rejection_sample(
        draft_tokens=draft_tokens,
        draft_logits=draft_logits,
        target_logits=target_logits,
        temperature=0.0
    )
    
    assert num_proposed == 4
    assert num_accepted_draft == 2
    assert accepted == [10, 20, 99]


def test_token_buffer_stop_detection():
    class MockTokenizer:
        def decode(self, tokens, **kwargs):
            return " ".join([f"w{t}" for t in tokens])
    
    tokenizer = MockTokenizer()
    buffer = TokenBuffer(tokenizer, stop_tokens=[999])
    buffer.initialize_prompt([1, 2, 3])
    
    valid, text, finished = buffer.append_and_check([10, 20, 999, 30])
    assert finished is True
    assert 999 not in valid
    assert valid == [10, 20]


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "target_model" in data
    assert "draft_model" in data


def test_models_endpoint():
    client = TestClient(app)
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 2


def test_chat_completions_non_streaming():
    client = TestClient(app)
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 20,
        "temperature": 0.7,
        "stream": False,
        "use_speculative": True
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert len(data["choices"][0]["message"]["content"]) > 0
    assert data["usage"]["completion_tokens"] > 0


def test_chat_completions_streaming():
    client = TestClient(app)
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 30,
        "stream": True,
        "use_speculative": True
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    
    lines = [line.strip() for line in response.text.split("\n") if line.startswith("data: ")]
    assert len(lines) > 1
    assert lines[-1] == "data: [DONE]"
