import asyncio
from typing import AsyncGenerator, List, Optional, Tuple, Dict, Any, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from app.config import settings
from app.utils.logger import logger
from app.utils.metrics import RequestTelemetry, update_gpu_metrics
from app.core.sampler import speculative_rejection_sample, sample_token
from app.core.cache import KVCacheManager, TokenBuffer


class SpeculativeEngine:
    def __init__(self):
        self.device = settings.resolved_device()
        self.dtype = settings.resolved_dtype()
        self.lookahead_k = settings.LOOKAHEAD_K
        
        self.target_model: Optional[PreTrainedModel] = None
        self.draft_model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.is_ready: bool = False
        self.use_mock: bool = settings.USE_MOCK_ENGINE

    def load_models(self):
        if self.use_mock:
            self._init_mock_models()
            self.is_ready = True
            return

        logger.info(
            "Loading target model '%s' and draft model '%s' on %s (%s)",
            settings.TARGET_MODEL_ID,
            settings.DRAFT_MODEL_ID,
            self.device,
            self.dtype
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.TARGET_MODEL_ID,
                trust_remote_code=True,
                padding_side="left"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.draft_model = AutoModelForCausalLM.from_pretrained(
                settings.DRAFT_MODEL_ID,
                torch_dtype=self.dtype,
                device_map=self.device,
                trust_remote_code=True
            ).eval()

            self.target_model = AutoModelForCausalLM.from_pretrained(
                settings.TARGET_MODEL_ID,
                torch_dtype=self.dtype,
                device_map=self.device,
                trust_remote_code=True
            ).eval()

            self.is_ready = True
            logger.info("Models loaded successfully")
            update_gpu_metrics()

        except Exception as e:
            logger.warning("Failed to load HuggingFace weights (%s). Falling back to mock engine.", e)
            self._init_mock_models()
            self.is_ready = True

    def _init_mock_models(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception:
            class MockTokenizer:
                vocab_size = 50257
                pad_token_id = 50256
                eos_token_id = 50256
                def encode(self, text, **kwargs):
                    return [hash(w) % 50000 + 1 for w in text.split()]
                def decode(self, tokens, **kwargs):
                    return " " + " ".join([f"t{t % 1000}" for t in tokens])
                def apply_chat_template(self, messages, **kwargs):
                    return "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            self.tokenizer = MockTokenizer()
        self.use_mock = True

    def format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and callable(self.tokenizer.apply_chat_template):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        
        formatted = ""
        for msg in messages:
            formatted += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        formatted += "<|im_start|>assistant\n"
        return formatted

    def tokenize(self, prompt: str) -> List[int]:
        if hasattr(self.tokenizer, "encode"):
            return self.tokenizer.encode(prompt, add_special_tokens=False)
        return [100, 200, 300]

    async def generate_speculative_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        k: Optional[int] = None,
        stop_tokens: Optional[List[int]] = None,
        stop_strings: Optional[List[str]] = None,
        telemetry: Optional[RequestTelemetry] = None
    ) -> AsyncGenerator[Tuple[str, List[int]], None]:
        if telemetry is None:
            telemetry = RequestTelemetry(mode="speculative")

        k_val = k or self.lookahead_k
        input_ids = self.tokenize(prompt)
        telemetry.prompt_tokens = len(input_ids)

        if self.use_mock:
            async for chunk, tokens in self._generate_mock_speculative(
                input_ids, max_tokens, k_val, telemetry
            ):
                yield chunk, tokens
            return

        stop_ids = list(stop_tokens or [])
        if self.tokenizer.eos_token_id is not None:
            stop_ids.append(self.tokenizer.eos_token_id)

        token_buffer = TokenBuffer(self.tokenizer, stop_tokens=stop_ids, stop_strings=stop_strings)
        token_buffer.initialize_prompt(input_ids)

        curr_input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        draft_cache = None
        target_cache = None

        with torch.inference_mode():
            target_out = self.target_model(curr_input_ids, use_cache=True)
            target_cache = target_out.past_key_values
            next_token_logits = target_out.logits[:, -1, :]
            
            first_token = sample_token(next_token_logits, temperature, top_p)
            valid_toks, text_chunk, is_finished = token_buffer.append_and_check([first_token])
            
            telemetry.record_first_token()
            telemetry.record_token_emitted(len(valid_toks))
            if text_chunk:
                yield text_chunk, valid_toks

            if is_finished or len(token_buffer.generated_token_ids) >= max_tokens:
                return

            while len(token_buffer.generated_token_ids) < max_tokens and not token_buffer.is_finished:
                draft_tokens: List[int] = []
                draft_logits_list: List[torch.Tensor] = []
                
                draft_input = torch.tensor([[token_buffer.all_token_ids[-1]]], dtype=torch.long, device=self.device)
                
                if draft_cache is None:
                    full_prefix = torch.tensor([token_buffer.all_token_ids], dtype=torch.long, device=self.device)
                    d_out = self.draft_model(full_prefix, use_cache=True)
                    draft_cache = d_out.past_key_values
                    d_logits = d_out.logits[:, -1, :]
                else:
                    d_out = self.draft_model(draft_input, past_key_values=draft_cache, use_cache=True)
                    draft_cache = d_out.past_key_values
                    d_logits = d_out.logits[:, -1, :]

                for _ in range(k_val):
                    d_tok = sample_token(d_logits, temperature, top_p)
                    draft_tokens.append(d_tok)
                    draft_logits_list.append(d_logits.squeeze(0))

                    next_d_input = torch.tensor([[d_tok]], dtype=torch.long, device=self.device)
                    d_out = self.draft_model(next_d_input, past_key_values=draft_cache, use_cache=True)
                    draft_cache = d_out.past_key_values
                    d_logits = d_out.logits[:, -1, :]

                draft_logits_tensor = torch.stack(draft_logits_list, dim=0)

                target_verify_input = torch.tensor([draft_tokens], dtype=torch.long, device=self.device)
                t_out = self.target_model(target_verify_input, past_key_values=target_cache, use_cache=True)
                target_eval_logits = t_out.logits.squeeze(0)

                accepted_tokens, num_accepted_draft, num_proposed = speculative_rejection_sample(
                    draft_tokens=draft_tokens,
                    draft_logits=draft_logits_tensor,
                    target_logits=target_eval_logits,
                    temperature=temperature,
                    top_p=top_p
                )

                telemetry.record_speculation_cycle(num_proposed, num_accepted_draft)

                actual_valid_seq_len = len(token_buffer.all_token_ids) + len(accepted_tokens)
                target_cache = KVCacheManager.crop_cache(t_out.past_key_values, actual_valid_seq_len)
                draft_cache = KVCacheManager.crop_cache(draft_cache, actual_valid_seq_len)

                valid_toks, text_chunk, is_finished = token_buffer.append_and_check(accepted_tokens)
                telemetry.record_token_emitted(len(valid_toks))

                if text_chunk:
                    yield text_chunk, valid_toks

                await asyncio.sleep(0)

                if is_finished or len(token_buffer.generated_token_ids) >= max_tokens:
                    break

    async def generate_baseline_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop_tokens: Optional[List[int]] = None,
        stop_strings: Optional[List[str]] = None,
        telemetry: Optional[RequestTelemetry] = None
    ) -> AsyncGenerator[Tuple[str, List[int]], None]:
        if telemetry is None:
            telemetry = RequestTelemetry(mode="vanilla")

        input_ids = self.tokenize(prompt)
        telemetry.prompt_tokens = len(input_ids)

        if self.use_mock:
            async for chunk, tokens in self._generate_mock_vanilla(input_ids, max_tokens, telemetry):
                yield chunk, tokens
            return

        stop_ids = list(stop_tokens or [])
        if self.tokenizer.eos_token_id is not None:
            stop_ids.append(self.tokenizer.eos_token_id)

        token_buffer = TokenBuffer(self.tokenizer, stop_tokens=stop_ids, stop_strings=stop_strings)
        token_buffer.initialize_prompt(input_ids)

        curr_input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        past_key_values = None

        with torch.inference_mode():
            while len(token_buffer.generated_token_ids) < max_tokens and not token_buffer.is_finished:
                if past_key_values is None:
                    out = self.target_model(curr_input_ids, use_cache=True)
                else:
                    out = self.target_model(curr_input_ids, past_key_values=past_key_values, use_cache=True)

                past_key_values = out.past_key_values
                next_logits = out.logits[:, -1, :]
                next_token = sample_token(next_logits, temperature, top_p)

                valid_toks, text_chunk, is_finished = token_buffer.append_and_check([next_token])
                
                if telemetry.first_token_time is None:
                    telemetry.record_first_token()
                telemetry.record_token_emitted(1)

                if text_chunk:
                    yield text_chunk, valid_toks

                curr_input_ids = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
                await asyncio.sleep(0)

                if is_finished:
                    break

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        use_speculative: bool = True,
        k: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None
    ) -> Tuple[str, RequestTelemetry]:
        mode = "speculative" if use_speculative else "vanilla"
        telemetry = RequestTelemetry(mode=mode)
        stop_strings = [stop] if isinstance(stop, str) else (stop or [])
        full_text = []

        if use_speculative:
            gen = self.generate_speculative_stream(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                k=k,
                stop_strings=stop_strings,
                telemetry=telemetry
            )
        else:
            gen = self.generate_baseline_stream(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_strings=stop_strings,
                telemetry=telemetry
            )

        async for chunk, _ in gen:
            full_text.append(chunk)

        return "".join(full_text), telemetry

    async def _generate_mock_speculative(
        self, input_ids: List[int], max_tokens: int, k_val: int, telemetry: RequestTelemetry
    ):
        mock_tokens = [
            "def", "binary_search", "(", "arr", ",", "target", ")", ":",
            "\n    ", "left", ",", "right", "=", "0", ",", "len", "(", "arr", ")", "-", "1",
            "\n    ", "while", "left", "<=", "right", ":",
            "\n        ", "mid", "=", "(", "left", "+", "right", ")", "//", "2",
            "\n        ", "if", "arr", "[", "mid", "]", "==", "target", ":",
            "\n            ", "return", "mid",
            "\n        ", "elif", "arr", "[", "mid", "]", "<", "target", ":",
            "\n            ", "left", "=", "mid", "+", "1",
            "\n        ", "else", ":",
            "\n            ", "right", "=", "mid", "-", "1",
            "\n    ", "return", "-", "1", "\n"
        ]
        tok_idx = 0
        while tok_idx < len(mock_tokens) and telemetry.completion_tokens < max_tokens:
            proposed = min(k_val, len(mock_tokens) - tok_idx)
            accepted = max(1, int(proposed * 0.75))
            chunk_tokens = mock_tokens[tok_idx : tok_idx + accepted]
            tok_idx += accepted
            text = " ".join(chunk_tokens) + " "
            tokens = [1000 + i for i in range(len(chunk_tokens))]

            telemetry.record_speculation_cycle(proposed, accepted)
            if telemetry.first_token_time is None:
                telemetry.record_first_token()
            telemetry.record_token_emitted(len(tokens))

            await asyncio.sleep(0.015)
            yield text, tokens

    async def _generate_mock_vanilla(
        self, input_ids: List[int], max_tokens: int, telemetry: RequestTelemetry
    ):
        mock_tokens = [
            "def", "binary_search", "(", "arr", ",", "target", ")", ":",
            "\n    ", "left", ",", "right", "=", "0", ",", "len", "(", "arr", ")", "-", "1",
            "\n    ", "while", "left", "<=", "right", ":",
            "\n        ", "mid", "=", "(", "left", "+", "right", ")", "//", "2",
            "\n        ", "if", "arr", "[", "mid", "]", "==", "target", ":",
            "\n            ", "return", "mid",
            "\n    ", "return", "-", "1", "\n"
        ]
        for word in mock_tokens:
            if telemetry.completion_tokens >= max_tokens:
                break
            text = word + " "
            tokens = [1000]
            if telemetry.first_token_time is None:
                telemetry.record_first_token()
            telemetry.record_token_emitted(1)
            await asyncio.sleep(0.015)
            yield text, tokens


engine = SpeculativeEngine()
