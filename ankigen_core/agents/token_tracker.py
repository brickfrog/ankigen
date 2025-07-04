"""
Token usage tracking for OpenAI API calls using tiktoken.
Provides accurate token counting and cost estimation.
"""

import tiktoken
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from ankigen_core.logging import logger


@dataclass
class TokenUsage:
    """Track token usage for a single request"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: Optional[float]
    model: str
    timestamp: datetime = field(default_factory=datetime.now)


class TokenTracker:
    """Track token usage across multiple requests"""

    def __init__(self):
        self.usage_history: List[TokenUsage] = []
        self.total_cost = 0.0
        self.total_tokens = 0

    def count_tokens_for_messages(
        self, messages: List[Dict[str, str]], model: str
    ) -> int:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("o200k_base")

        tokens_per_message = 3
        tokens_per_name = 1

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(str(value)))
                if key == "name":
                    num_tokens += tokens_per_name

        num_tokens += 3
        return num_tokens

    def count_tokens_for_text(self, text: str, model: str) -> int:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("o200k_base")

        return len(encoding.encode(text))

    def estimate_cost(
        self, prompt_tokens: int, completion_tokens: int, model: str
    ) -> Optional[float]:
        return None

    def track_usage_from_response(
        self, response_data, model: str
    ) -> Optional[TokenUsage]:
        try:
            if hasattr(response_data, "usage"):
                usage = response_data.usage
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens

                actual_cost = None
                if hasattr(usage, "total_cost"):
                    actual_cost = usage.total_cost
                elif hasattr(usage, "cost"):
                    actual_cost = usage.cost

                return self.track_usage(
                    prompt_tokens, completion_tokens, model, actual_cost
                )
            return None
        except Exception as e:
            logger.error(f"Failed to track usage from response: {e}")
            return None

    def track_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        actual_cost: Optional[float] = None,
    ) -> TokenUsage:
        total_tokens = prompt_tokens + completion_tokens

        if actual_cost is not None:
            final_cost = actual_cost
        else:
            final_cost = self.estimate_cost(prompt_tokens, completion_tokens, model)

        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost=final_cost,
            model=model,
        )

        self.usage_history.append(usage)
        if final_cost:
            self.total_cost += final_cost
        self.total_tokens += total_tokens

        logger.info(
            f"💰 Token usage - Model: {model}, Prompt: {prompt_tokens}, Completion: {completion_tokens}, Cost: ${final_cost:.4f}"
            if final_cost
            else f"💰 Token usage - Model: {model}, Prompt: {prompt_tokens}, Completion: {completion_tokens}"
        )

        return usage

    def get_session_summary(self) -> Dict[str, Any]:
        if not self.usage_history:
            return {
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "by_model": {},
            }

        by_model = {}
        for usage in self.usage_history:
            if usage.model not in by_model:
                by_model[usage.model] = {"requests": 0, "tokens": 0, "cost": 0.0}
            by_model[usage.model]["requests"] += 1
            by_model[usage.model]["tokens"] += usage.total_tokens
            if usage.estimated_cost:
                by_model[usage.model]["cost"] += usage.estimated_cost

        return {
            "total_requests": len(self.usage_history),
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "by_model": by_model,
        }

    def get_session_usage(self) -> Dict[str, Any]:
        return self.get_session_summary()

    def reset_session(self):
        self.usage_history.clear()
        self.total_cost = 0.0
        self.total_tokens = 0
        logger.info("🔄 Token usage tracking reset")

    def track_usage_from_agents_sdk(
        self, usage_dict: Dict[str, Any], model: str
    ) -> Optional[TokenUsage]:
        """Track usage from OpenAI Agents SDK usage format"""
        try:
            if not usage_dict or usage_dict.get("total_tokens", 0) == 0:
                return None

            prompt_tokens = usage_dict.get("input_tokens", 0)
            completion_tokens = usage_dict.get("output_tokens", 0)

            return self.track_usage(prompt_tokens, completion_tokens, model)
        except Exception as e:
            logger.error(f"Failed to track usage from agents SDK: {e}")
            return None


# Global token tracker instance
_global_tracker = TokenTracker()


def get_token_tracker() -> TokenTracker:
    return _global_tracker


def track_agent_usage(
    prompt_text: str,
    completion_text: str,
    model: str,
    actual_cost: Optional[float] = None,
) -> TokenUsage:
    tracker = get_token_tracker()

    prompt_tokens = tracker.count_tokens_for_text(prompt_text, model)
    completion_tokens = tracker.count_tokens_for_text(completion_text, model)

    return tracker.track_usage(prompt_tokens, completion_tokens, model, actual_cost)


def track_usage_from_openai_response(response_data, model: str) -> Optional[TokenUsage]:
    tracker = get_token_tracker()
    return tracker.track_usage_from_response(response_data, model)


def track_usage_from_agents_sdk(
    usage_dict: Dict[str, Any], model: str
) -> Optional[TokenUsage]:
    """Track usage from OpenAI Agents SDK usage format"""
    tracker = get_token_tracker()
    return tracker.track_usage_from_agents_sdk(usage_dict, model)
