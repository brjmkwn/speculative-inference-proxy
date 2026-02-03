from typing import Optional, Any, List, Tuple
import torch
from transformers.cache_utils import DynamicCache


class KVCacheManager:
    @staticmethod
    def crop_cache(past_key_values: Any, target_length: int) -> Any:
        if past_key_values is None:
            return None

        if isinstance(past_key_values, DynamicCache):
            if hasattr(past_key_values, "crop"):
                past_key_values.crop(target_length)
                return past_key_values
            
            for layer_idx in range(len(past_key_values.key_cache)):
                k = past_key_values.key_cache[layer_idx]
                v = past_key_values.value_cache[layer_idx]
                if k is not None and k.shape[-2] > target_length:
                    past_key_values.key_cache[layer_idx] = k[..., :target_length, :]
                    past_key_values.value_cache[layer_idx] = v[..., :target_length, :]
            return past_key_values

        if isinstance(past_key_values, (tuple, list)):
            new_cache = []
            for layer in past_key_values:
                if layer is None:
                    continue
                k, v = layer[0], layer[1]
                if k.shape[-2] > target_length:
                    k = k[..., :target_length, :]
                    v = v[..., :target_length, :]
                new_cache.append((k, v))
            return tuple(new_cache)

        return past_key_values

    @staticmethod
    def get_seq_length(past_key_values: Any) -> int:
        if past_key_values is None:
            return 0
        if isinstance(past_key_values, DynamicCache):
            return past_key_values.get_seq_length()
        if isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
            return past_key_values[0][0].shape[-2]
        return 0


class TokenBuffer:
    def __init__(self, tokenizer: Any, stop_tokens: Optional[List[int]] = None, stop_strings: Optional[List[str]] = None):
        self.tokenizer = tokenizer
        self.stop_tokens = set(stop_tokens or [])
        self.stop_strings = stop_strings or []
        self.all_token_ids: List[int] = []
        self.generated_token_ids: List[int] = []
        self.emitted_text_len: int = 0
        self.is_finished: bool = False
        self.finish_reason: Optional[str] = None

    def initialize_prompt(self, prompt_tokens: List[int]):
        self.all_token_ids = list(prompt_tokens)

    def append_and_check(self, new_tokens: List[int]) -> Tuple[List[int], str, bool]:
        valid_tokens: List[int] = []

        for token in new_tokens:
            if self.is_finished:
                break

            if token in self.stop_tokens:
                self.is_finished = True
                self.finish_reason = "stop"
                break

            self.all_token_ids.append(token)
            self.generated_token_ids.append(token)
            valid_tokens.append(token)

        current_full_text = self.tokenizer.decode(self.generated_token_ids, skip_special_tokens=True)
        new_text = current_full_text[self.emitted_text_len:]
        self.emitted_text_len = len(current_full_text)

        if self.stop_strings and not self.is_finished:
            for s in self.stop_strings:
                if s in current_full_text:
                    self.is_finished = True
                    self.finish_reason = "stop"
                    stop_idx = current_full_text.find(s)
                    new_text = current_full_text[self.emitted_text_len - len(new_text):stop_idx]
                    break

        return valid_tokens, new_text, self.is_finished
