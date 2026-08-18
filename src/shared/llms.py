from openai import OpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


def is_rate_limited(e):
    return (
        getattr(e, "code", None) == 429
        or "429" in str(e)
        or "RESOURCE_EXHAUSTED" in str(e)
    )


_retry = retry(
    retry=retry_if_exception(is_rate_limited),
    wait=wait_exponential(multiplier=1, min=20, max=120),
    stop=stop_after_attempt(5),
)


class DeepSeekProvider:
    def __init__(self, *, api_key: str | None = None, model: str = "deepseek-chat"):
        self._client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self._model = model

    @_retry
    def generate_text(self, prompt: str) -> str:
        r = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8192,
            temperature=0,
        )
        if not r.choices or not r.choices[0].message:
            return ""
        return r.choices[0].message.content or ""
