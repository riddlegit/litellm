import asyncio
import builtins
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import litellm
from litellm.integrations.headroom import HeadroomLogger


# ---------------- helpers ----------------


def _make_result(messages=None, tokens_saved=100, before=1000, after=900):
    return SimpleNamespace(
        messages=(
            messages
            if messages is not None
            else [{"role": "user", "content": "compressed"}]
        ),
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


# ---------------- async_pre_call_hook: success + guards ----------------


def _data(messages=None):
    return {
        "model": "gpt-4o",
        "messages": (
            messages
            if messages is not None
            else [{"role": "user", "content": "hello world"}]
        ),
    }


def test_hook_compresses_and_replaces_messages(monkeypatch):
    _set_token_count(monkeypatch, 1000)  # above default min_tokens (500)
    compressed = [{"role": "user", "content": "compressed"}]
    compress_mock = MagicMock(
        return_value=_make_result(messages=compressed, tokens_saved=100)
    )
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


# ---------------- no-op when compression saves nothing (coverage) ----------------


def test_hook_no_replace_when_tokens_saved_zero(monkeypatch):
    _set_token_count(monkeypatch, 1000)
    original_messages = [{"role": "user", "content": "hello world"}]
    compress_mock = MagicMock(
        return_value=_make_result(
            messages=[{"role": "user", "content": "x"}], tokens_saved=0
        )
    )
    _install_fake_compress(monkeypatch, compress_mock)

    logger = HeadroomLogger()
    data = _data(messages=original_messages)
    result = _run_hook(logger, data)

    assert compress_mock.called
    assert (
        result["messages"] == original_messages
    )  # NOT replaced when tokens_saved == 0
    assert logger.total_tokens_saved == 0


# ---------------- string-callback instantiation (the real enablement path) ----------------


def test_string_callback_instantiates_to_logger():
    """callbacks: ['headroom'] must instantiate a HeadroomLogger via the
    hardcoded if/elif resolver (not just the registry). Regression for the bug
    where the short name resolved to None and the feature was silently inert."""
    from litellm.litellm_core_utils.litellm_logging import (
        _init_custom_logger_compatible_class,
        get_custom_logger_compatible_class,
    )

    inst = _init_custom_logger_compatible_class("headroom", None, None)
    assert isinstance(inst, HeadroomLogger)

    looked_up = get_custom_logger_compatible_class("headroom")
    assert isinstance(looked_up, HeadroomLogger)
