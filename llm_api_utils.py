import random
import cohere
import os
import openai
from pathlib import Path
import time
import backoff


PRICE_PER_1K_TOKENS = {
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-4-1106-preview": {"prompt": 0.01, "completion": 0.03},
    "gpt-3.5-turbo": {"prompt": 0.001, "completion": 0.002},
    "ada": {"embed": 0.0004},
    "text-embedding-ada-002": {"embed": 0.0001},
}


EMBEDDING_DIMENSIONS = {
    "ada": 1536,
    "text-embedding-ada-002": 1536,
}


def estimate_cost_of_text_generation_api_call(
    model: str, response: dict, verbose: bool
) -> float:
    completion_tokens = response.usage.completion_tokens
    prompt_tokens = response.usage.prompt_tokens
    total_tokens = response.usage.total_tokens

    prompt_cost = prompt_tokens / 1000 * PRICE_PER_1K_TOKENS[model]["prompt"]
    completion_cost = (
        completion_tokens / 1000 * PRICE_PER_1K_TOKENS[model]["completion"]
    )
    cost = prompt_cost + completion_cost

    if verbose:
        summary = f"""\
Used {prompt_tokens} prompt tokens, {completion_tokens} completion tokens, {total_tokens} total tokens
Esimated cost: {cost:.4f} USD
"""
        print(summary)
    return cost


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIConnectionError))
def call_openai_with_exponetial_backoff(**kwargs):
    rand_sleep_in_secs = 5 * random.random()
    time.sleep(rand_sleep_in_secs)
    return openai.chat.completions.create(**kwargs)


def init_openai_with_api_key():
    openai.api_key = os.environ.get("OPENAI_API_KEY")


def init_cohere_with_api_key():
    COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
    co = cohere.Client(COHERE_API_KEY)
    return co
