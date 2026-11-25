# Model Setup

## Agents And Shared Resources
Models are chosen per **agent**, not globally. Open **Agents** to see your agents as cards: each shows its status (active, locked, archived, default), version and which workflows it supports (DILI Assessment, DILI Revision, Timeline). Icon buttons (hover or focus for the tooltip) edit, duplicate, set as default, lock or unlock, archive or restore, and delete an agent; a disabled button's tooltip explains why. **New agent** and **Edit** open the same page: the **Agent profile** card (name, description, lifecycle) beside a placeholder **Agent validation** card, then one collapsible section per workflow, each with its own on/off switch, models, instructions and settings. Model choices sit in a shared **Model(s)** section with one **Refresh models** action; each model carries its own reasoning level (**Model default** sends no reasoning preference), and catalog warnings appear on a line below the controls. DILI Revision has a single optional **Custom instructions** field (review focus, writing style, terminology, tone) and one model used for every review stage; its stages, limits and tools are fixed by the application. Timeline has only provider, model and reasoning, because extraction is scripted. The DILI Assessment report template is set with a **Write | Upload file** control. Turning a workflow off keeps its settings but makes the agent unavailable for that workflow. **Save agent** creates a new numbered revision; earlier sessions keep the revision they ran with. Locked, inactive and archived agents are read-only until unlocked, reactivated or restored from the profile card. The default agent cannot be deactivated, archived or deleted, and an agent used by saved runs can only be archived, not deleted.

**Settings → Providers & Models** is shared by every agent: access keys, provider model catalogs (install or refresh), and the knowledge-index (RAG) settings, which autosave. Per-agent model pickers use these catalogs.

Last updated: 2026-11-25

Sampling settings are selected automatically according to the provider, model
family, and operation. They are not user-configurable.

## Configure Models
1. Add and activate a provider access key in **Settings → Providers & Models** if a
   hosted provider is used, then load or refresh that provider's catalog.
2. Open **Agents**, edit the default agent or create one, and pick a catalog model
   for each role of each enabled workflow. A model may hold more than one role.
3. Save the agent. Each save creates a new revision; runs keep the revision they
   started with.

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
1. Open **Settings → Providers & Models**.
2. Choose the provider.
3. Open the access-key dialog or management control.
4. Paste the provider key.
5. Save it. New keys remain inactive, so a rejected key cannot replace the current active key.
6. Explicitly activate the key that should be used.
7. Confirm the active-key indicator is shown.
8. Return to **Settings → Providers & Models** and use **Refresh** for the selected provider when its catalog needs updating.

Do not paste keys into screenshots, chat messages, issue reports, or shared logs.

For OpenCode Go, refresh the provider catalog after provider model changes. The
application preserves the selected model and reports the provider's sanitized
HTTP detail if structured extraction is rejected; it does not silently switch
to another model or provider.

## Local Model Notes
- Agent model pickers list only installed Ollama models for the local provider.
- A run uses exactly the model its agent revision names. If that Ollama model is
  no longer installed, the run fails with a message naming the model; it does not
  switch to another installed model.
- When installed, `qwen3.5:2b` is a fast local extractor and `qwen3.5:9b` is the
  stronger local backup option for bounded extraction tests. Note that the
  validated timeline boundary is model-specific: exact `qwen3.5:2b` does not
  produce grounded structured timeline output and fail-closes to a
deterministic fallback, while exact `qwen3.5:9b` does. See the [validation
ledger](../runtime/validation_ledger.md) for the supported local boundary.
