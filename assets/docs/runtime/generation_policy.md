# Effective LLM inference policy

Last updated: 2026-08-21

DILIGENT resolves an effective inference configuration immediately before each
LLM request. Operators choose the provider, model, and one global reasoning
level: `off`, `low`, `medium`, or `high`. The selected level is persisted as
`reasoning_level`. A legacy persisted `ollama_reasoning=false` is read as
`off`, and `true` is read as `medium`; new writes use only `reasoning_level`.

## Purpose responsibility

The global preference is transformed into a task-level request so structured
work does not inherit the same reasoning cost as clinical synthesis.

| Purpose | Off | Low | Medium | High |
| --- | --- | --- | --- | --- |
| Clinical synthesis | off | low | medium | high |
| Structured extraction | off | low | low | low |
| Faithful rewrite | off | low | low | low |
| Revision scan/planning/tool selection/editing/QA | off | low | low | medium |
| Timeline, simple | off | low | low | low |
| Timeline, moderate | off | low | medium | medium |
| Timeline, complex | off | low | medium | high |
| JSON repair/connectivity | off | off | off | off |

The policy records both the user-requested and purpose-requested levels. The
capability resolver then records the provider-effective level and an explicit
coercion reason when a model cannot honor the request.

## Sampling and output budgets

The catalog at `app/resources/catalogs/llm_generation_policies.json` defines
the deterministic base temperatures: clinical synthesis `0.2`; extraction,
timeline, revision, and faithful rewrite `0.0`; JSON repair and connectivity
omit temperature. Capability rules may omit temperature when the selected
model does not support sampling controls or when active reasoning makes the
provider parameter invalid. Caller options cannot override the effective
configuration.

The capability catalog at
`app/resources/catalogs/llm_model_capabilities.json` resolves exact model,
longest family prefix, provider, then conservative fallback. Context capacity
is the intersection of catalog/model capacity and live local runtime capacity.
The input budget is that capacity less visible output, reasoning reserve, and
safety reserve. Capacity is a ceiling, not a target: context segments are
deduplicated, prioritized, and reported when omitted or overflowing.

## Provider normalization

OpenAI Responses preserves normalized options and uses `max_output_tokens`;
OpenAI Chat Completions uses `max_tokens`; Anthropic reserves a separate
thinking budget; Gemini maps the four levels to its supported thinking levels;
and Ollama sends `think=false`, `think=true`, or the GPT-OSS level string as
supported by the selected model. Unknown or unsupported reasoning is visible
in the effective configuration rather than silently treated as enabled.

The policy version, purpose, provider, model, requested/effective levels,
temperature, capability source, reserves, input budget, and compact context
selection report are captured in runtime provenance.

When adding a model, update both catalogs, verify the primary vendor contract,
add resolver and provider-payload tests, and update the source review date.
