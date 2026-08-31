# Model Setup
Last updated: 2026-08-31

Sampling settings are selected automatically according to the provider, model
family, and operation. They are not user-configurable.

## Configure Models
Open **Configurations** from the sidebar.

This page controls how the application calls a model during analysis. Expect controls for:
- local provider configuration
- cloud provider configuration
- model selection
- saving or applying model settings
- provider access-key management where credentials are required

Recommended workflow:
1. Decide whether the run should use a local or cloud provider.
2. For local testing, select an Ollama-compatible model if available.
3. Select a catalog model independently for the **Clinical**, **Text extraction**,
   **Revision**, and **Timeline** roles. A model may hold more than one role.
4. Save or apply the configuration.
5. Add and activate an access key if the provider requires one.

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
1. Open **Configurations**.
2. Choose the provider.
3. Open the access-key dialog or management control.
4. Paste the provider key.
5. Save it. New keys remain inactive, so a rejected key cannot replace the current active key.
6. Explicitly activate the key that should be used.
7. Confirm the active-key indicator is shown.
8. Return to **Configurations** and use **Refresh** for the selected provider when its catalog needs updating.

Do not paste keys into screenshots, chat messages, issue reports, or shared logs.

## Local Model Notes
- Local runtime saves only installed Ollama models for clinical, extraction,
  Revision, and Timeline roles.
- Revision and Timeline workflows show the configured role model and link back to
  this page; they do not contain per-run provider/model selectors.
- If you switch from cloud to local mode, cloud-only role selections are cleared automatically.
- When installed, `qwen3.5:2b` is the preferred fast local extractor and `qwen3.5:9b` is the recommended stronger backup option for bounded extraction tests.
