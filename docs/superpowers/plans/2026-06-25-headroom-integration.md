# Headroom Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native Headroom (LLM message compression) integration to litellm as a first-class proxy callback, togglable via `litellm_settings: callbacks: ["headroom"]`, using local in-process compression.

**Architecture:** A new `HeadroomLogger(CustomLogger)` in `litellm/integrations/headroom.py` overrides `async_pre_call_hook` (proxy path). It filters to `completion`/`acompletion` calls, pre-filters small messages via `litellm.token_counter`, lazily imports `headroom.compress.compress`, replaces `data["messages"]` with the compressed result, and safely returns the original `data` on any failure. It is registered under the short name `"headroom"` in the callback registry and `__init__.py` Literal so `callbacks: ["headroom"]` resolves to it.

**Tech Stack:** Python, Pydantic v2, pytest (sync tests using `asyncio.run`), litellm `CustomLogger` base, optional `headroom-ai` package (lazy-imported).

**Spec:** `docs/superpowers/specs/2026-06-25-headroom-integration-design.md`

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `litellm/integrations/headroom.py` | `HeadroomLogger` class — the entire integration (config, hook, lazy compress loader) | Create |
| `litellm/litellm_core_utils/custom_logger_registry.py` | Map short name `"headroom"` → `HeadroomLogger` | Modify |
| `litellm/__init__.py` | Register `"headroom"` in the callback Literal | Modify |
| `litellm/integrations/callback_configs.json` | UI dashboard metadata entry for headroom | Modify |
| `litellm/proxy/example_config_yaml/headroom_config.yaml` | Example proxy config showing how to enable | Create |
| `tests/test_litellm/test_headroom_integration.py` | Unit tests (all mocked; no real `headroom-ai` needed) | Create |

**Design notes for the implementer:**
- `headroom-ai` is **not** installed in dev/CI and is **never** imported at module load — only lazily inside `_load_compress()`. So `import litellm.integrations.headroom` always succeeds.
- The proxy invokes `async_pre_call_hook` with **keyword args** `user_api_key_dict=, cache=, data=, call_type=`. Parameter *names* must match exactly; types can be `Any`.
- Tests inject a fake `headroom.compress` module into `sys.modules` and patch `litellm.token_counter` — no network, no `headroom-ai`.

---

## Task 1: `HeadroomLogger` skeleton + `__init__` config parsing

**Files:**
- Create: `litellm/integrations/headroom.py`
- Test: `tests/test_litellm/test_headroom_integration.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_litellm/test_headroom_integration.py`:

```python
import asyncio
import builtins
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import litellm
from litellm.integrations.headroom import HeadroomLogger


# ---------------- helpers ----------------

def _make_result(messages=None, tokens_saved=100, before=1000, after=900):
    return SimpleNamespace(
        messages=messages
        if messages is not None
        else [{"role": "user", "content": "compressed"}],
        tokens_before=before,
        tokens_after=after,
        tokens_saved=tokens_saved,
        compression_ratio=after / before,
    )


def _install_fake_compress(monkeypatch, compress):
    """Inject a fake headroom.compress module so `from headroom.compress import compress` resolves."""
    fake_pkg = types.ModuleType("headroom")
    fake_mod = types.ModuleType("headroom.compress")
    fake_mod.compress = compress
    monkeypatch.setitem(sys.modules, "headroom", fake_pkg)
    monkeypatch.setitem(sys.modules, "headroom.compress", fake_mod)


def _set_token_count(monkeypatch, count):
    """Make litellm.token_counter report a fixed token count for messages."""
    monkeypatch.setattr(litellm, "token_counter", lambda **kwargs: count)


def _run_hook(logger, data, call_type="completion"):
    return asyncio.run(
        logger.async_pre_call_hook(
            user_api_key_dict=None, cache=None, data=data, call_type=call_type
        )
    )


# ---------------- __init__ config ----------------

def test_init_defaults():
    logger = HeadroomLogger()
    assert logger.min_tokens == 500
    assert logger.model_limit == 200000
    assert logger.total_tokens_saved == 0
    assert logger._compress_fn is None
    assert logger._import_failed is False


def test_init_reads_env_vars(monkeypatch):
    monkeypatch.setenv("HEADROOM_MIN_TOKENS", "1234")
    monkeypatch.setenv("HEADROOM_MODEL_LIMIT", "9999")
    logger = HeadroomLogger()
    assert logger.min_tokens == 1234
    assert logger.model_limit == 9999


def test_init_explicit_args_override_env(monkeypatch):
    monkeypatch.setenv("HEADROOM_MIN_TOKENS", "1234")
    logger = HeadroomLogger(min_tokens=42, model_limit=8)
    assert logger.min_tokens == 42
    assert logger.model_limit == 8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_litellm/test_headroom_integration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'litellm.integrations.headroom'` (or import error).

- [ ] **Step 3: Write minimal implementation**

Create `litellm/integrations/headroom.py`:

```python
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

    def _load_compress(self) -> Optional[Any]:
        if self._compress_fn is None:
            from headroom.compress import compress

            self._compress_fn = compress
        return self._compress_fn
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_litellm/test_headroom_integration.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add litellm/integrations/headroom.py tests/test_litellm/test_headroom_integration.py
git commit -m "feat(headroom): add HeadroomLogger skeleton with env-var config

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Register short name `"headroom"` in registry + Literal

**Files:**
- Modify: `litellm/litellm_core_utils/custom_logger_registry.py`
- Modify: `litellm/__init__.py`
- Test: `tests/test_litellm/test_headroom_integration.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_litellm/test_headroom_integration.py`:

```python
# ---------------- registry short-name resolution ----------------

def test_registry_resolves_headroom():
    from litellm.litellm_core_utils.custom_logger_registry import CustomLoggerRegistry

    assert (
        CustomLoggerRegistry.get_class_type_for_custom_logger_name("headroom")
        is HeadroomLogger
    )
    assert (
        CustomLoggerRegistry.get_callback_str_from_class_type(HeadroomLogger)
        == "headroom"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_litellm/test_headroom_integration.py::test_registry_resolves_headroom -v`
Expected: FAIL with `KeyError: 'headroom'` (not yet in the registry dict).

- [ ] **Step 3: Register in the callback registry**

In `litellm/litellm_core_utils/custom_logger_registry.py`:

Add the import (alphabetically between the `gitlab` and `humanloop` imports):

```python
from litellm.integrations.gitlab import GitLabPromptManager
from litellm.integrations.headroom import HeadroomLogger
from litellm.integrations.humanloop import HumanloopLogger
```

Add the dict entry inside `CALLBACK_CLASS_STR_TO_CLASS_TYPE` (right after the `"focus"` entry):

```python
        "focus": FocusLogger,
        "headroom": HeadroomLogger,
        "vantage": VantageLogger,
```

- [ ] **Step 4: Register in the `__init__.py` Literal**

In `litellm/__init__.py`, add `"headroom",` to the `_custom_logger_compatible_callbacks_literal` list (at the end, after `"compression_interception",`):

```python
    "levo",
    "compression_interception",
    "headroom",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_litellm/test_headroom_integration.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add litellm/litellm_core_utils/custom_logger_registry.py litellm/__init__.py tests/test_litellm/test_headroom_integration.py
git commit -m "feat(headroom): register 'headroom' short name in callback registry

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `async_pre_call_hook` — happy path + guards

This task implements the compression flow for the success paths and the skip-guards (non-completion call type, empty messages, below `min_tokens`). Failure safety (try/except, missing-package caching) is added in Task 4.

**Files:**
- Modify: `litellm/integrations/headroom.py` (add `async_pre_call_hook`)
- Test: `tests/test_litellm/test_headroom_integration.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_litellm/test_headroom_integration.py`:

```python
# ---------------- async_pre_call_hook: success + guards ----------------

def _data(messages=None):
    return {
        "model": "gpt-4o",
        "messages": messages
        if messages is not None
        else [{"role": "user", "content": "hello world"}],
    }


def test_hook_compresses_and_replaces_messages(monkeypatch):
    _set_token_count(monkeypatch, 1000)  # above default min_tokens (500)
    compressed = [{"role": "user", "content": "compressed"}]
    compress_mock = MagicMock(return_value=_make_result(messages=compressed, tokens_saved=100))
    _install_fake_compress(monkeypatch, compress_mock)

    logger = HeadroomLogger()
    data = _data()
    result = _run_hook(logger, data)

    assert compress_mock.called
    assert result["messages"] == compressed
    assert logger.total_tokens_saved == 100


def test_hook_accumulates_tokens_saved(monkeypatch):
    _set_token_count(monkeypatch, 1000)
    compress_mock = MagicMock(return_value=_make_result(tokens_saved=50))
    _install_fake_compress(monkeypatch, compress_mock)

    logger = HeadroomLogger()
    _run_hook(logger, _data())
    _run_hook(logger, _data())
    assert logger.total_tokens_saved == 100


def test_hook_skips_non_completion_call_type(monkeypatch):
    compress_mock = MagicMock(return_value=_make_result())
    _install_fake_compress(monkeypatch, compress_mock)

    logger = HeadroomLogger()
    data = _data()
    result = _run_hook(logger, data, call_type="embeddings")

    assert not compress_mock.called
    assert result is data  # unchanged, same object


def test_hook_skips_empty_messages(monkeypatch):
    _set_token_count(monkeypatch, 1000)
    compress_mock = MagicMock(return_value=_make_result())
    _install_fake_compress(monkeypatch, compress_mock)

    logger = HeadroomLogger()
    data = {"model": "gpt-4o", "messages": []}
    result = _run_hook(logger, data)

    assert not compress_mock.called
    assert result is data


def test_hook_skips_below_min_tokens(monkeypatch):
    _set_token_count(monkeypatch, 10)  # below default min_tokens (500)
    compress_mock = MagicMock(return_value=_make_result())
    _install_fake_compress(monkeypatch, compress_mock)

    logger = HeadroomLogger()
    data = _data()
    result = _run_hook(logger, data)

    assert not compress_mock.called
    assert result is data
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_litellm/test_headroom_integration.py -v`
Expected: The 5 new tests FAIL. `test_hook_compresses_and_replaces_messages` fails because `async_pre_call_hook` is the base-class no-op (returns `None`), so `result["messages"]` raises `TypeError`. The skip tests fail because compress is never invoked by the no-op base, but `result is data` fails since base returns `None` (not `data`).

- [ ] **Step 3: Implement `async_pre_call_hook`**

In `litellm/integrations/headroom.py`, add this method to the `HeadroomLogger` class (after `__init__`, before `_load_compress`):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_litellm/test_headroom_integration.py -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add litellm/integrations/headroom.py tests/test_litellm/test_headroom_integration.py
git commit -m "feat(headroom): compress messages in async_pre_call_hook

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Safe fallback on failure + missing-package caching

Wrap the hook body in `try/except` so any error returns the original `data`, and cache the `headroom-ai` import failure so it isn't retried on every request.

**Files:**
- Modify: `litellm/integrations/headroom.py` (wrap hook in try/except; harden `_load_compress`)
- Test: `tests/test_litellm/test_headroom_integration.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_litellm/test_headroom_integration.py`:

```python
# ---------------- failure safety ----------------

def _block_headroom_import(monkeypatch):
    """Make `import headroom...` deterministically fail even if headroom-ai is installed.
    Returns a list whose [0] is the number of import attempts observed."""
    calls = [0]
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "headroom" or name.startswith("headroom."):
            calls[0] += 1
            raise ImportError("simulated: headroom-ai not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "headroom", raising=False)
    monkeypatch.delitem(sys.modules, "headroom.compress", raising=False)
    return calls


def test_hook_falls_back_when_compress_raises(monkeypatch):
    _set_token_count(monkeypatch, 1000)
    compress_mock = MagicMock(side_effect=RuntimeError("compress boom"))
    _install_fake_compress(monkeypatch, compress_mock)

    logger = HeadroomLogger()
    data = _data()
    original_messages = data["messages"]
    result = _run_hook(logger, data)  # must NOT raise

    assert result["messages"] == original_messages  # original messages preserved
    assert logger.total_tokens_saved == 0


def test_hook_falls_back_when_headroom_missing(monkeypatch):
    _set_token_count(monkeypatch, 1000)
    _block_headroom_import(monkeypatch)

    logger = HeadroomLogger()
    data = _data()
    original_messages = data["messages"]
    result = _run_hook(logger, data)  # must NOT raise

    assert result["messages"] == original_messages
    assert logger.total_tokens_saved == 0
    assert logger._import_failed is True


def test_hook_does_not_retry_import_after_failure(monkeypatch):
    _set_token_count(monkeypatch, 1000)
    import_calls = _block_headroom_import(monkeypatch)

    logger = HeadroomLogger()
    _run_hook(logger, _data())
    calls_after_first = import_calls[0]
    _run_hook(logger, _data())
    _run_hook(logger, _data())

    assert import_calls[0] == calls_after_first  # no additional import attempts
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_litellm/test_headroom_integration.py -v`
Expected: The 3 new tests FAIL. `test_hook_falls_back_when_compress_raises` raises `RuntimeError` (no try/except yet). `test_hook_falls_back_when_headroom_missing` raises `ImportError` from `_load_compress`. `test_hook_does_not_retry_import_after_failure` raises `ImportError`.

- [ ] **Step 3: Wrap the hook body in try/except**

In `litellm/integrations/headroom.py`, replace the body of `async_pre_call_hook` (from the `token_count = ...` line through `return data`) so the compression work is inside a `try/except`. The full method becomes:

```python
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
```

- [ ] **Step 4: Harden `_load_compress` with import-failure caching**

In `litellm/integrations/headroom.py`, replace the `_load_compress` method with:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_litellm/test_headroom_integration.py -v`
Expected: PASS (12 tests).

- [ ] **Step 6: Commit**

```bash
git add litellm/integrations/headroom.py tests/test_litellm/test_headroom_integration.py
git commit -m "feat(headroom): safe fallback on compression/import failure

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: UI metadata + example proxy config

**Files:**
- Modify: `litellm/integrations/callback_configs.json`
- Create: `litellm/proxy/example_config_yaml/headroom_config.yaml`

- [ ] **Step 1: Add the headroom entry to `callback_configs.json`**

In `litellm/integrations/callback_configs.json`, the array currently ends with the SQS entry followed by `]`:

```json
    "description": "SQS Queue (AWS) Logging Integration"
  }
]
```

Add a trailing comma after the SQS entry's closing `}` and append a new headroom object, so the end becomes:

```json
    "description": "SQS Queue (AWS) Logging Integration"
  },
  {
    "id": "headroom",
    "displayName": "Headroom",
    "logo": "headroom.png",
    "supports_key_team_logging": false,
    "dynamic_params": {
      "headroom_min_tokens": {
        "type": "number",
        "ui_name": "Min Tokens",
        "description": "Skip compression for requests below this token count (default 500)",
        "required": false
      },
      "headroom_model_limit": {
        "type": "number",
        "ui_name": "Model Token Limit",
        "description": "Model context limit passed to Headroom (default 200000)",
        "required": false
      }
    },
    "description": "Headroom message compression integration (local mode). Requires pip install headroom-ai"
  }
]
```

> Note: the primary configuration channel is environment variables (`HEADROOM_MIN_TOKENS`, `HEADROOM_MODEL_LIMIT`); the `dynamic_params` here are dashboard UI hints. No `headroom.png` logo exists in the repo — the UI renders gracefully without it; do not commit a binary asset without a source.

- [ ] **Step 2: Verify the JSON is valid**

Run: `python -c "import json; json.load(open('litellm/integrations/callback_configs.json')); print('valid')"`
Expected: prints `valid`.

- [ ] **Step 3: Create the example proxy config**

Create `litellm/proxy/example_config_yaml/headroom_config.yaml`:

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

litellm_settings:
  # Enable Headroom message compression (remove this line to disable).
  callbacks: ["headroom"]

# Headroom tuning (environment variables):
#   HEADROOM_MIN_TOKENS=500      # skip compression for requests below this token count
#   HEADROOM_MODEL_LIMIT=200000  # model context limit passed to Headroom
#
# Requires the optional package: pip install headroom-ai
```

- [ ] **Step 4: Verify the YAML parses**

Run: `python -c "import yaml; yaml.safe_load(open('litellm/proxy/example_config_yaml/headroom_config.yaml')); print('valid')"`
Expected: prints `valid`. (If `pyyaml` is not installed, run `uv run python ...` instead.)

- [ ] **Step 5: Commit**

```bash
git add litellm/integrations/callback_configs.json litellm/proxy/example_config_yaml/headroom_config.yaml
git commit -m "feat(headroom): add UI metadata and example proxy config

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Format, full test run, lint

**Files:** none (verification only)

- [ ] **Step 1: Format with Black**

Run: `uv run black litellm/integrations/headroom.py tests/test_litellm/test_headroom_integration.py`
Expected: files reformatted (or already formatted). Re-run to confirm `left unchanged`.

- [ ] **Step 2: Run the headroom tests**

Run: `uv run pytest tests/test_litellm/test_headroom_integration.py -v`
Expected: PASS (12 tests).

- [ ] **Step 3: Run Ruff on the new files**

Run: `uv run ruff check litellm/integrations/headroom.py tests/test_litellm/test_headroom_integration.py`
Expected: `All checks passed!`

- [ ] **Step 4: Run the broader unit suite to catch regressions**

Run: `uv run pytest tests/test_litellm/ -x -q 2>&1 | tail -30`
Expected: no failures related to the registry / Literal change. (If a pre-existing unrelated failure appears, confirm it is unrelated by checking it on `main`.)

- [ ] **Step 5: Final commit if Black changed anything**

```bash
git status --short
# if headroom.py / test file changed:
git add litellm/integrations/headroom.py tests/test_litellm/test_headroom_integration.py
git commit -m "style(headroom): black formatting

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Manual smoke test (optional, requires `headroom-ai`)

If `headroom-ai` is installed, verify end-to-end through the registry path:

```bash
uv run python -c "
import asyncio, litellm
from litellm.litellm_core_utils.custom_logger_registry import CustomLoggerRegistry
H = CustomLoggerRegistry.get_class_type_for_custom_logger_name('headroom')()
litellm.callbacks = [H]
print('registered:', type(H).__name__)
"
```

Expected: prints `registered: HeadroomLogger` with no import errors.

---

## Definition of Done

- `callbacks: ["headroom"]` resolves to `HeadroomLogger` via the registry (Task 2 test).
- `async_pre_call_hook` compresses completion messages, skips non-completions / empty / small messages, and safely falls back on any error or missing `headroom-ai` (Tasks 3–4 tests).
- UI metadata and example config are valid JSON/YAML (Task 5).
- `black` + `ruff` clean; `tests/test_litellm/test_headroom_integration.py` all pass; no regression in the unit suite (Task 6).
