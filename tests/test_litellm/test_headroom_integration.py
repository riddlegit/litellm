import asyncio
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
