"""
Headroom integration — compresses messages via Headroom before each proxy
completion call (local / in-process mode).

Enable in proxy config:
    litellm_settings:
      callbacks: ["headroom"]

Tune via environment variables:
    HEADROOM_MIN_TOKENS   (default 500)    — skip compression below this token count
    HEADROOM_MODEL_LIMIT  (default 200000) — model context limit passed to headroom

Requires the optional `headroom-ai` package: pip install headroom-ai
"""

import os
from typing import Any, Optional

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger

_DEFAULT_MIN_TOKENS = 500
_DEFAULT_MODEL_LIMIT = 200000
_DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
_COMPRESSION_CALL_TYPES = ("completion", "acompletion")


class HeadroomLogger(CustomLogger):
    """LiteLLM proxy callback that compresses messages via Headroom before each
    completion call.

    Implements ``async_pre_call_hook`` (proxy path). Returns the request ``data``
    with ``data["messages"]`` replaced by the compressed messages. On any failure
    (token counting, missing ``headroom-ai`` package, compression error) it
    safely returns the original ``data`` unchanged — it never breaks the request.
    """

    def __init__(
        self,
        min_tokens: Optional[int] = None,
        model_limit: Optional[int] = None,
        hooks: Any = None,
    ) -> None:
        super().__init__()
        self.min_tokens = (
            min_tokens
            if min_tokens is not None
            else int(os.getenv("HEADROOM_MIN_TOKENS", str(_DEFAULT_MIN_TOKENS)))
        )
        self.model_limit = (
            model_limit
            if model_limit is not None
            else int(os.getenv("HEADROOM_MODEL_LIMIT", str(_DEFAULT_MODEL_LIMIT)))
        )
        self.hooks = hooks
        self.total_tokens_saved = 0
        self._compress_fn: Optional[Any] = None
        self._import_failed = False

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: Any,
    ) -> dict:
        # Only compress chat completion requests.
        if str(call_type) not in _COMPRESSION_CALL_TYPES:
            return data

        messages = data.get("messages") or []
        if not messages:
            return data
        model = data.get("model", "") or ""

        token_count = litellm.token_counter(model=model, messages=messages)
        if token_count < self.min_tokens:
            return data

        compress_fn = self._load_compress()
        if compress_fn is None:
            return data

        result = compress_fn(
            messages=messages,
            model=model or _DEFAULT_MODEL,
            model_limit=self.model_limit,
            hooks=self.hooks,
        )

        if result is not None and getattr(result, "tokens_saved", 0) > 0:
            data["messages"] = result.messages
            self.total_tokens_saved += result.tokens_saved
            verbose_logger.info(
                "Headroom: %s->%s tokens (saved %s) [total saved: %s]",
                getattr(result, "tokens_before", "?"),
                getattr(result, "tokens_after", "?"),
                result.tokens_saved,
                self.total_tokens_saved,
            )

        return data

    def _load_compress(self) -> Optional[Any]:
        if self._compress_fn is None:
            from headroom.compress import compress

            self._compress_fn = compress
        return self._compress_fn
