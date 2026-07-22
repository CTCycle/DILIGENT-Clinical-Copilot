# DILI end-to-end validation
Date: 2026-07-22

## Executive summary

A dummy DILI case was created and completed successfully through the clinical-job API. The saved session demonstrates deterministic DILI analysis, an evidence-backed local LiverTox match, manual report editing, and timeline persistence. A revision-start defect found during the run was repaired and regression-tested.

The requested OpenCode Go / DeepSeek Flash configuration was selected and active, but live calls to `opencode.ai` consistently failed with connection errors. The clinical job therefore completed using deterministic fallbacks rather than provider-generated content. The RAG-on path cannot perform retrieval because no Ollama embedding model is configured. The in-app browser could not attach a tab and its first fallback browser executable was absent; Chrome subsequently provided live frontend evidence. Responsive and keyboard navigation coverage is still outstanding.

## Test scope and results

| Area | Result | Evidence |
| --- | --- | --- |
| OpenCode Go / DeepSeek Flash | Partial | Config persisted `opencode_go` and `deepseek-v4-flash`; active OpenCode key is present. Provider connections failed. |
| Dummy DILI case, RAG off | Pass with provider fallback | Job `91ce8149` completed; persisted session `3` has a hepatocellular assessment and `use_rag: false`. |
| RAG on | Blocked by configuration | Preflight reported `rag_embedding_model_missing`, allowed continuation, and correctly disclosed no configured Ollama embedding model. |
| Reasoning on/off | Pass | UI slider changed from Medium to Off and back to Medium; the UI reported that extra parameters were saved. Backend boolean state was also restored to true. |
| Access-key activation | Pass | The OpenCode key modal showed a masked active key and last-used timestamp. Addition of a second valid key was not tested because no spare secret was supplied. |
| Manual text editing | Pass | Report edit created an auditable newer version with prior/new hashes and reviewer metadata. |
| Timeline generation | Pass with provider fallback | Valid cloud override persisted timeline `7`, `source_model: deepseek-v4-flash`, `model_provider: opencode_go`, `generation_status: fallback`. |
| Session revision | Fixed and rechecked | Revision initially returned 500; after repair it returned 202 and job `986f3612` started. It then failed separately because revision defaulted to unavailable local Ollama. |
| Case interruption | Pass | Job `3c57e02a` was cancelled via documented DELETE; final status `cancelled` at 23%. |
| Input validation | Pass | Short cancellation input received 422 with `clinical_input must contain at least 60 words`; valid canonical three-section input passed preflight. |
| Frontend initial load and configuration UI | Pass | Chrome rendered the DILI input page and loaded the configured OpenCode Go / DeepSeek Flash catalog without console warnings or errors. |
| Responsive and keyboard audit | Unverified | No smaller-viewport or full keyboard-navigation pass was completed. |

## Findings

### High — OpenCode generation is unreachable

The configured provider/model is active, but every live provider request failed to connect to `opencode.ai`. The primary DILI case completed only because the workflow fell back to deterministic extraction and local evidence. This affects clinical synthesis and timeline extraction quality.

Reproduction: select OpenCode Go with `deepseek-v4-flash`, submit a valid DILI case, then inspect the application log. It records repeated `Cloud LLM call failed: All connection attempts failed` messages.

### Medium — RAG-on is not operational

RAG preflight correctly warns that no Ollama embedding model is configured and permits the explicitly disclosed no-RAG fallback. Retrieval cannot be tested until an embedding model is configured and reachable.

### Medium — Revision start broke after manual editing (fixed)

Manual editing creates a later session version. Revision lookup assumed a single version for the same session and raised `MultipleResultsFound`, returning a generic 500.

Repair: revision lookup now selects the latest version deterministically; initial-version synchronization explicitly selects version 1. Regression coverage includes a manual edit followed by revision job creation.

### Medium — Reasoning persistence has two representations

The UI exposes Off/Low/Medium/High and successfully auto-saved a Medium-to-Off-to-Medium cycle, whereas the documented model-config API exposes only `ollama_reasoning: true|false`. The active UI behavior is correct for this run, but API-level auditing cannot distinguish Low, Medium, and High.

## Artifacts and cleanup

The intentionally retained dummy session is `QA Dummy DILI Case 20260722` (session `3`) and contains a visible QA manual-edit marker. The cancellation-only case did not persist as a completed session. No real patient data was used.

## Follow-up: OpenCode Go credential and routing validation

The supplied OpenCode credential was added, activated, and deduplicated so that exactly one masked OpenCode key remains active. No secret value is recorded in this report.

The runtime test first exposed a routing defect: the current OpenCode Go model catalogue returns `deepseek-v4-flash` without an endpoint declaration, while the application required that declaration. The transport now routes that documented Go model to `chat/completions` (and documented Anthropic-family Go models to `messages`). Unit coverage passed: `15 passed`.

With the repaired backend running outside the restricted test sandbox, its connectivity check reaches the documented `https://opencode.ai/zen/go/v1/chat/completions` endpoint. The provider responds `401 Unauthorized` for the active supplied credential. Therefore this is no longer a network or in-app endpoint-routing failure; a valid OpenCode Go-enabled credential is required before a provider-generated DILI run can pass. RAG validation is intentionally out of scope for this follow-up.

### Credential replacement follow-up

A second supplied credential was stored and activated, replacing the previous inactive credential so that one masked OpenCode credential remains active. The repaired connectivity check again reached the documented Go endpoint and again received `401 Unauthorized`. This confirms that neither supplied credential is currently authorized for the OpenCode Go API from this application.

## Recommended next actions

1. Provide an OpenCode Go-enabled API key, then rerun the same dummy case and verify actual provider provenance rather than fallback artifacts.
2. Define a provider-agnostic reasoning-level contract if API-level distinction between Low, Medium, and High is required for cloud models.
3. Complete a smaller-viewport, keyboard-navigation, and network-inspection UI pass.
