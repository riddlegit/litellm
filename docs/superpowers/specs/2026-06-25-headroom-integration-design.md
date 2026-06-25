# Headroom 原生集成 — 设计文档

- **日期**: 2026-06-25
- **分支**: `feat/headroom-integration`
- **状态**: 待评审

## 1. 目标

在 litellm 中新增一个原生的 **Headroom** 集成，作为 proxy 的一等公民 callback。Headroom 在消息发给 LLM provider 之前对其进行压缩，从而节省 token / 成本。集成可通过 proxy 配置参数打开和关闭。

- **参考文档**: https://headroom-docs.vercel.app/docs/litellm
- **参考源码**: https://github.com/chopratejas/headroom/blob/main/headroom/integrations/litellm_callback.py

## 2. 背景

Headroom 有两种压缩模式：

- **Local 模式**：进程内压缩，调用 `headroom.compress.compress(messages, model, model_limit, hooks)`，返回带 `.messages / .tokens_before / .tokens_after / .tokens_saved / .compression_ratio` 的对象。需要安装 `headroom-ai` 包。
- **Cloud 模式**：调用 Headroom Cloud API，需要 `HEADROOM_API_KEY` + httpx。

上游的 `HeadroomCallback` 实现了 LiteLLM 的 `CustomLogger` 接口，但存在一个关键问题：它的 `async_pre_call_hook` 签名 `(user_api_key, data, call_type)` 与 LiteLLM 真实的 `CustomLogger.async_pre_call_hook` 签名 `(user_api_key_dict, cache, data, call_type)` 不一致；且其文档宣称「SDK 下 `litellm.callbacks=[HeadroomCallback()]` 自动压缩」，但经核实 `async_pre_call_hook` **只在 proxy 路径被调用**，SDK 路径（`litellm.completion`）走的是另一个 hook —— 因此上游 callback 实际只在 proxy 生效。做一个**原生的、签名正确的** litellm 集成正是本次工作的价值。

## 3. 关键决策

| # | 决策点 | 选择 |
|---|---|---|
| 1 | 压缩模式 | **仅 Local 模式**。进程内压缩，懒加载 `headroom.compress()`；不引入 Cloud / httpx / API key。需要用户另装 `headroom-ai`。 |
| 2 | 开关机制 | **短名注册 + 环境变量**。在 callback registry 与 `__init__.py` Literal 注册短名 `"headroom"`；通过 `litellm_settings: callbacks: ["headroom"]` 开关；调参用环境变量 `HEADROOM_MIN_TOKENS` / `HEADROOM_MODEL_LIMIT`。 |
| 3 | 改动范围 | **完整集成**：新模块 + registry + Literal + `callback_configs.json` UI 元数据 + 示例 yaml + 单测。 |
| 4 | `min_tokens` 行为 | **加 token 预过滤**。压缩前用 `litellm.token_counter` 估算消息 token 数，低于 `HEADROOM_MIN_TOKENS`（默认 500）的请求跳过压缩（让参数真正生效，优于上游存而不用）。 |
| 5 | Hook 选择 | **`async_pre_call_hook`（proxy-only）**。见第 4 节。 |

## 4. 架构：Hook 选择

LiteLLM 有两个「请求发出前改消息」的 hook：

- `async_pre_call_hook(self, user_api_key_dict, cache, data, call_type) -> Optional[Union[Exception, str, dict]]`：**仅 proxy 路径**调用（`litellm/proxy/utils.py:1433`）。返回 dict 会经 `process_pre_call_hook_response` 替换发给 provider 的 `data`。是 proxy 请求改写的标准 hook（guardrail 同款）。proxy 对 `litellm.callbacks` 中所有直接覆盖了该方法的 `CustomLogger` 子类都会调用。
- `async_pre_call_deployment_hook(self, kwargs, call_type) -> Optional[dict]`：**SDK 路径**调用（`litellm/utils.py:1273`，经 `utils.py:1812` 的 completion 包装器），返回 dict 替换 `kwargs`。

**选择 `async_pre_call_hook`（仅实现这一个）的理由：**

1. 本集成的开关机制 `litellm_settings.callbacks` 是 **proxy 配置**，目标环境是 proxy。
2. 这是 proxy 请求改写的官方、最可靠的 hook。
3. **避免双重压缩**：proxy 请求会先经过 `async_pre_call_hook`，随后 proxy → router → `litellm.acompletion` 又会触发 deployment hook。若两个 hook 都实现，同一请求会被压缩两次。只实现 `async_pre_call_hook`，deployment hook 走基类 no-op，无双重压缩。

**后果（已接受的取舍）**：直接把 litellm 当 SDK 调用（`litellm.completion(...)` 而不经 proxy）时不会压缩。这与上游 callback 的实际行为一致（尽管其文档另有宣称）。SDK 直接用法不在本次范围。

## 5. 组件设计：`HeadroomLogger`

新文件 `litellm/integrations/headroom.py`，单文件单类（对标 `lago.py` / `langsmith.py`）。

```python
import os
from typing import Any, Optional, Union

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger


class HeadroomLogger(CustomLogger):
    """LiteLLM proxy callback that compresses messages via Headroom before each
    completion call (local / in-process mode).

    Enable in proxy config:
        litellm_settings:
          callbacks: ["headroom"]
    Tune via env: HEADROOM_MIN_TOKENS (default 500), HEADROOM_MODEL_LIMIT (default 200000).
    Requires the optional `headroom-ai` package: pip install headroom-ai
    """

    def __init__(
        self,
        min_tokens: Optional[int] = None,
        model_limit: Optional[int] = None,
        hooks: Any = None,
    ) -> None:
        super().__init__()
        self.min_tokens = min_tokens if min_tokens is not None else int(
            os.getenv("HEADROOM_MIN_TOKENS", "500")
        )
        self.model_limit = model_limit if model_limit is not None else int(
            os.getenv("HEADROOM_MODEL_LIMIT", "200000")
        )
        self.hooks = hooks
        self.total_tokens_saved = 0
        self._compress_fn = None  # 懒加载 headroom.compress.compress
        self._import_failed = False

    async def async_pre_call_hook(
        self, user_api_key_dict, cache, data, call_type
    ):
        # 1) 只处理 completion / acompletion（兼容 string 与 enum）
        if str(call_type) not in ("completion", "acompletion"):
            return data

        messages = data.get("messages") or []
        if not messages:
            return data
        model = data.get("model", "") or ""

        try:
            # 2) min_tokens 预过滤
            token_count = litellm.token_counter(model=model, messages=messages)
            if token_count < self.min_tokens:
                return data

            # 3) 懒加载 headroom.compress
            compress_fn = self._load_compress()
            if compress_fn is None:
                return data  # import 失败已记 error，安全回退

            # 4) 压缩
            result = compress_fn(
                messages=messages,
                model=model or "claude-sonnet-4-5-20250929",
                model_limit=self.model_limit,
                hooks=self.hooks,
            )
            if result is not None and getattr(result, "tokens_saved", 0) > 0:
                data["messages"] = result.messages
                self.total_tokens_saved += result.tokens_saved
                verbose_logger.info(
                    "Headroom: %d→%d tokens (saved %d) [total saved: %d]",
                    result.tokens_before,
                    result.tokens_after,
                    result.tokens_saved,
                    self.total_tokens_saved,
                )
        except Exception as e:
            verbose_logger.warning(
                "Headroom compression failed, using original messages: %s", e
            )

        return data

    def _load_compress(self):
        if self._import_failed:
            return None
        if self._compress_fn is None:
            try:
                from headroom.compress import compress
                self._compress_fn = compress
            except Exception as e:
                self._import_failed = True
                verbose_logger.error(
                    "Headroom requires the `headroom-ai` package: pip install headroom-ai (%s)", e
                )
                return None
        return self._compress_fn
```

**设计要点：**

- **安全回退**：任何异常（token 计数、import、compress 抛错）都 `return data`（原消息），绝不中断用户请求。
- **懒加载 + 失败缓存**：`headroom` 首次调用时 import；失败则置 `_import_failed=True`，后续请求直接回退、不再重复 import，避免每请求一次无谓的 ImportError。
- **min_tokens 预过滤**：`litellm.token_counter(model, messages)` 返回 int，未知模型不抛错（用默认 token 参数）。低于阈值跳过。
- **call_type 容错**：`str(call_type) in ("completion","acompletion")`，兼容字符串与 enum。
- **可观测**：用 `verbose_logger` 记录压缩前后 token 数与累计节省。

## 6. 新建 / 修改文件

| 操作 | 文件 | 说明 |
|---|---|---|
| 新建 | `litellm/integrations/headroom.py` | `HeadroomLogger` 类 |
| 改 | `litellm/litellm_core_utils/custom_logger_registry.py` | 顶部 import（`from litellm.integrations.headroom import HeadroomLogger`，按字母序插在 `gitlab` 与 `humanloop` 之间）；`CALLBACK_CLASS_STR_TO_CLASS_TYPE` 加 `"headroom": HeadroomLogger`（该 dict 非严格排序，插在 `"focus"` 附近即可） |
| 改 | `litellm/__init__.py` | `_custom_logger_compatible_callbacks_literal` 加 `"headroom",`（该 Literal 非严格字母序；位置不影响功能，加在列表末尾 `compression_interception` 后即可） |
| 改 | `litellm/integrations/callback_configs.json` | 加 headroom 条目（displayName / description / dynamic_params：`headroom_min_tokens`、`headroom_model_limit`）。主配置渠道仍是环境变量，dynamic_params 仅作 dashboard 提示。 |
| 新建 | `litellm/proxy/example_config_yaml/headroom_config.yaml` | 示例配置 |
| 新建 | `tests/test_litellm/test_headroom_integration.py` | 单元测试 |

> **logo 资源**：本仓库 `ui/litellm-dashboard/public/assets/logos/` 没有 `headroom.png`。UI 会优雅缺图，不阻塞功能。本次不引入二进制图片资源（避免无来源的资产）。

## 7. 配置示例

```yaml
# litellm_config.yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-xxx

litellm_settings:
  callbacks: ["headroom"]   # 列上=开启，删掉=关闭
```

```bash
# 环境变量（或 .env）
export HEADROOM_MIN_TOKENS=500
export HEADROOM_MODEL_LIMIT=200000
pip install headroom-ai     # 必需的本地压缩依赖
```

## 8. 测试计划

`tests/test_litellm/test_headroom_integration.py`，全部用 monkeypatch mock，不依赖真实 `headroom-ai`：

1. **正常压缩**：mock `headroom.compress.compress` 返回 `tokens_saved>0`，断言 `async_pre_call_hook` 返回的 `data["messages"]` 被替换、`total_tokens_saved` 累加。
2. **min_tokens 预过滤**：mock `token_counter` 返回 < `min_tokens`，断言不调用 compress、原样返回 `data`。
3. **call_type 过滤**：`call_type="embeddings"` 断言不压缩、原样返回。
4. **compress 抛异常**：mock compress 抛错，断言安全回退原 messages、不抛出。
5. **headroom 未安装**：让 `from headroom.compress import compress` 抛 ImportError，断言安全回退、记 error、不抛出；再次调用不重复 import。
6. **环境变量解析**：设 `HEADROOM_MIN_TOKENS=1234` / `HEADROOM_MODEL_LIMIT=9999`，断言 `__init__` 正确解析。
7. **registry 短名解析**：`CustomLoggerRegistry.get_class_type_for_custom_logger_name("headroom")` 返回 `HeadroomLogger`；`get_callback_str_from_class_type(HeadroomLogger)` 返回 `"headroom"`。
8. **空 messages**：`data["messages"]=[]` 断言原样返回、不压缩。

**质量门槛**：`uv run black .` 格式化；`make test-unit`（或 `uv run pytest tests/test_litellm/test_headroom_integration.py -v`）通过。

## 9. 范围外（YAGNI）

- Cloud 模式（Headroom Cloud API / `HEADROOM_API_KEY`）。
- SDK 直接用法（`litellm.completion` 不经 proxy）的压缩。
- logo 矢量图资产。
- prompt management。
- per-key / per-team 的动态 min_tokens / model_limit（当前是全局环境变量）。

## 10. 风险与备注

- **签名正确性**：`async_pre_call_hook` 必须用 `(self, user_api_key_dict, cache, data, call_type)` 签名，且直接定义在 `HeadroomLogger` 类上（proxy 通过 `"async_pre_call_hook" in vars(_callback.__class__)` 判断是否覆盖）。
- **双重压缩**：已通过「只实现 `async_pre_call_hook`」规避。
- **可选依赖**：`headroom-ai` 不写入 litellm 的核心依赖；用户自行安装。import 失败时优雅降级。
- **token_counter 性能**：每个 completion 请求多一次本地 token 计数（开销小，纯本地估算）。
