# Model Setup

## Agents And Shared Resources
Models are chosen per **agent**, not globally. Open **Agent Studio** (**My Agents**) to see your agents as cards: each shows its status (active, locked, archived, default), version and which workflows it supports (DILI Assessment, DILI Revision, Timeline). Icon buttons (hover or focus for the tooltip) edit, duplicate, set as default, lock or unlock, archive or restore, and delete an agent; a disabled button's tooltip explains why. **New agent** and **Edit** open the same page: the **Agent profile** card (name, description, lifecycle) beside a placeholder **Agent validation** card, then one collapsible section per workflow, each with its own on/off switch, models, instructions and settings. Turning a workflow off keeps its settings but makes the agent unavailable for that workflow. **Save agent** creates a new numbered revision; earlier sessions keep the revision they ran with. Locked, inactive and archived agents are read-only until unlocked, reactivated or restored from the profile card. The default agent cannot be deactivated, archived or deleted, and an agent used by saved runs can only be archived, not deleted.

**Providers & Resources** is shared by every agent: access keys, provider model catalogs (install or refresh), and the knowledge-index (RAG) settings, which autosave. Per-agent model pickers use these catalogs.

Last updated: 2026-11-12

Sampling settings are selected automatically according to the provider, model
family, and operation. They are not user-configurable.

## Configure Models
Open **Settings** from the sidebar and choose the **Models** page.

This page controls how the application calls a model during analysis. Expect controls for:
- local provider configuration
- cloud provider configuration
- model selection
- automatically saved model settings
- provider access-key management where credentials are required

Recommended workflow:
1. Decide whether the run should use a local or cloud provider.
2. For local testing, select an Ollama-compatible model if available.
3. Select a catalog model independently for the **Clinical**, **Text extraction**,
   **Revision**, and **Timeline** roles. A model may hold more than one role.
4. Valid model assignments, reasoning preferences, and RAG settings save automatically
   after a short pause. If a provider change clears the role assignments, select
   a model for each role to complete and save the configuration.
5. Add and activate an access key if the provider requires one. Key creation and
   activation remain explicit actions in the access-key controls.

## Manage Access Keys
Supported provider key operations currently cover:
- OpenAI
- Gemini
- DeepSeek
- Anthropic Claude
- OpenCode (shared by Zen and Go)
- Brave

Expected behavior:
- The application stores provider keys through the backend access-key service.
- The UI should show fingerprints and metadata rather than the full secret after saving.
- Only one key should be active for a provider at a time.
- Cloud and Ollama catalogs are saved in the application database. Opening the page reuses the saved catalog and does not contact a provider again. Use **Refresh** when you explicitly want a new provider listing. If a refresh fails, the last valid catalog remains visible; an empty Ollama installation is saved as an empty catalog. Catalog state is scoped by provider endpoint and active credential fingerprint, so changing credentials or endpoints starts a new cache scope without exposing secrets.

Recommended workflow:
1. Open **Agent Studio → Providers & Resources**.
2. Choose the provider.
3. Open the access-key dialog or management control.
4. Paste the provider key.
5. Save it. New keys remain inactive, so a rejected key cannot replace the current active key.
6. Explicitly activate the key that should be used.
7. Confirm the active-key indicator is shown.
8. Return to **Agent Studio → Providers & Resources** and use **Refresh** for the selected provider when its catalog needs updating.

Do not paste keys into screenshots, chat messages, issue reports, or shared logs.

For OpenCode Go, refresh the provider catalog after provider model changes. The
application preserves the selected model and reports the provider's sanitized
HTTP detail if structured extraction is rejected; it does not silently switch
to another model or provider.

## Local Model Notes
- Local runtime saves only installed Ollama models for clinical, extraction,
  Revision, and Timeline roles.
- Revision and Timeline workflows show the configured role model and link back to
  this page; they do not contain per-run provider/model selectors.
- If you switch from cloud to local mode, cloud-only role selections are cleared automatically.
- When installed, `qwen3.5:2b` is a fast local extractor and `qwen3.5:9b` is the
  stronger local backup option for bounded extraction tests. Note that the
  validated timeline boundary is model-specific: exact `qwen3.5:2b` does not
  produce grounded structured timeline output and fail-closes to a
deterministic fallback, while exact `qwen3.5:9b` does. See the [validation
ledger](../runtime/validation_ledger.md) for the supported local boundary.
