from typing import Tuple, List
import torch
import torch.nn.functional as F


def apply_temperature_and_top_p(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf")
) -> torch.Tensor:
    if temperature <= 0.0:
        return logits

    scaled_logits = logits / temperature

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        scaled_logits = scaled_logits.masked_fill(indices_to_remove, filter_value)

    return scaled_logits


def sample_token(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float = 1.0
) -> int:
    if logits.ndim > 1:
        logits = logits.squeeze(0)

    if temperature <= 1e-5:
        return int(torch.argmax(logits, dim=-1).item())

    processed_logits = apply_temperature_and_top_p(logits, temperature=temperature, top_p=top_p)
    probs = F.softmax(processed_logits, dim=-1)
    
    if torch.isnan(probs).any() or probs.sum() <= 0:
        return int(torch.argmax(logits, dim=-1).item())

    sampled = torch.multinomial(probs, num_samples=1)
    return int(sampled.item())


def speculative_rejection_sample(
    draft_tokens: List[int],
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float = 1.0
) -> Tuple[List[int], int, int]:
    """
    Evaluates draft tokens against target logits.
    Returns (accepted_tokens, num_accepted_draft, num_proposed_draft).
    """
    k = len(draft_tokens)
    accepted_tokens: List[int] = []
    num_accepted_draft = 0

    if temperature <= 1e-5:
        for i in range(k):
            target_argmax = int(torch.argmax(target_logits[i], dim=-1).item())
            draft_token = draft_tokens[i]

            if draft_token == target_argmax:
                accepted_tokens.append(draft_token)
                num_accepted_draft += 1
            else:
                accepted_tokens.append(target_argmax)
                return accepted_tokens, num_accepted_draft, k

        # All K tokens accepted; sample bonus token from position K
        bonus_token = int(torch.argmax(target_logits[k], dim=-1).item())
        accepted_tokens.append(bonus_token)
        return accepted_tokens, num_accepted_draft, k

    else:
        for i in range(k):
            draft_token = draft_tokens[i]
            
            p_draft_logits = apply_temperature_and_top_p(draft_logits[i], temperature, top_p)
            p_target_logits = apply_temperature_and_top_p(target_logits[i], temperature, top_p)
            
            p_draft = F.softmax(p_draft_logits, dim=-1)
            p_target = F.softmax(p_target_logits, dim=-1)

            q_x = float(p_draft[draft_token].item())
            p_x = float(p_target[draft_token].item())

            acceptance_prob = min(1.0, p_x / max(q_x, 1e-12))
            uniform_rand = float(torch.rand(1).item())

            if uniform_rand <= acceptance_prob:
                accepted_tokens.append(draft_token)
                num_accepted_draft += 1
            else:
                residual = torch.clamp(p_target - p_draft, min=0.0)
                residual_sum = torch.sum(residual)

                if residual_sum > 1e-8:
                    residual_probs = residual / residual_sum
                    correction_token = int(torch.multinomial(residual_probs, num_samples=1).item())
                else:
                    correction_token = int(torch.multinomial(p_target, num_samples=1).item())

                accepted_tokens.append(correction_token)
                return accepted_tokens, num_accepted_draft, k

        target_bonus_logits = apply_temperature_and_top_p(target_logits[k], temperature, top_p)
        target_bonus_probs = F.softmax(target_bonus_logits, dim=-1)
        bonus_token = int(torch.multinomial(target_bonus_probs, num_samples=1).item())
        accepted_tokens.append(bonus_token)

        return accepted_tokens, num_accepted_draft, k
