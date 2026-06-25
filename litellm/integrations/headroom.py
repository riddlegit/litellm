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

    def _load_compress(self) -> Optional[Any]:
        if self._compress_fn is None:
            from headroom.compress import compress

            self._compress_fn = compress
        return self._compress_fn
