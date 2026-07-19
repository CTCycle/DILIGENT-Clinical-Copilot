# Automatic generation policy

Last updated: 2026-07-19

DILIGENT selects sampling behavior automatically immediately before an LLM
request. Operators configure providers, models, reasoning, and retrieval only.
Temperature is not an operator, deployment, API, or per-run setting.

Resolution is deterministic: exact model, model family, catalogued local model
profile, provider compatibility, then the provider/model default. A `null`
temperature means the provider payload omits the parameter. Caller options are
sanitized and cannot override the policy.

The policy version is recorded in run provenance together with parser and
clinical policy IDs, purpose, provider, model, match kind, and effective
temperature or `provider_default`.

The current policy uses `0.0`/`0.2` as DILIGENT defaults for ordinary supported
instruction models, not as vendor guarantees. Qwen, DeepSeek-R1, Phi-4
Reasoning, Gemini, Anthropic, GPT-5, and GPT-OSS follow their documented model
compatibility behavior in the source-controlled catalog. Unsupported and
unknown models use the model/provider default.

When adding a model, add or update the catalog rule, verify the primary vendor
recommendation, add a resolver test, run provider payload tests, and update the
source review date. Do not add unrelated sampling controls such as `top_p`,
penalties, or a user override.
