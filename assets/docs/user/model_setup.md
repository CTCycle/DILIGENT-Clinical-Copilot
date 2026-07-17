# Model Setup
Last updated: 2026-07-17

## Configure Models
Open **Model Configurations** from the sidebar.

This page controls how the application calls a model during analysis. Expect controls for:
- local provider configuration
- cloud provider configuration
- model selection
- saving or applying model settings
- provider access-key management where credentials are required

Recommended workflow:
1. Decide whether the run should use a local or cloud provider.
2. For local testing, select an Ollama-compatible model if available.
3. For cloud use, select the intended provider and assign a catalog model to the clinical and extraction roles.
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
- Cloud model catalogs are loaded from each provider's official API after its key is active. If a later refresh fails, the page marks the last successful catalog as cached; after a backend restart, add or reactivate the key to refresh it again.

Recommended workflow:
1. Open **Model Configurations**.
2. Choose the provider.
3. Open the access-key dialog or management control.
4. Paste the provider key.
5. Save it.
6. Activate the key that should be used.
7. Confirm the active-key indicator is shown.

Do not paste keys into screenshots, chat messages, issue reports, or shared logs.

## Local Model Notes
- Local runtime saves only installed Ollama models for clinical and extraction roles.
- If you switch from cloud to local mode, cloud-only role selections are cleared automatically.
- When installed, `qwen3.5:2b` is the preferred fast local extractor and `qwen3.5:9b` is the recommended stronger backup option for bounded extraction tests.
