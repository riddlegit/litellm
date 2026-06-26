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


def _get_int_env(name: str, default: int) -> int:
    """Read an integer env var. Falls back to ``default`` (with a warning) if the
    variable is unset, empty, or non-numeric — so a malformed value never breaks
    callback initialization."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        verbose_logger.warning(
            "Headroom: invalid integer for %s=%r; using default %d",
            name,
            raw,
            default,
        )
        return default


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
            else _get_int_env("HEADROOM_MIN_TOKENS", _DEFAULT_MIN_TOKENS)
        )
        self.model_limit = (
            model_limit
            if model_limit is not None
            else _get_int_env("HEADROOM_MODEL_LIMIT", _DEFAULT_MODEL_LIMIT)
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

        try:
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
        except Exception as e:
            verbose_logger.warning(
                "Headroom compression failed, using original messages: %s", e
            )

        return data

    def _load_compress(self) -> Optional[Any]:
        """Lazily import ``headroom.compress.compress``. Returns None (and caches
        the failure) if ``headroom-ai`` is not installed."""
        if self._import_failed:
            return None
        if self._compress_fn is None:
            try:
                from headroom.compress import compress

                self._compress_fn = compress
            except Exception as e:
                self._import_failed = True
                verbose_logger.error(
                    "Headroom requires the `headroom-ai` package "
                    "(pip install headroom-ai). Skipping compression: %s",
                    e,
                )
                return None
        return self._compress_fn
