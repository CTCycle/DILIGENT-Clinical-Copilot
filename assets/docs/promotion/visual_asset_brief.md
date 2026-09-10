# Visual Asset Brief
Last updated: 2026-09-10

## Objective
Create promotional visuals that make DILIGENT understandable in a few seconds without implying autonomous diagnosis or showing sensitive clinical information.

The visual story should emphasize three ideas:

1. structured DILI review;
2. deterministic and inspectable safeguards;
3. configurable local or cloud LLM assistance.

## Source Material
Prefer existing synthetic-data screenshots already maintained in the repository. The release screenshot set is available under:

`assets/QA/release-v3.3.0-screenshots/`

The README currently uses release views for:

- DILI Agent;
- Clinical Sessions;
- Patient Timeline;
- Data Inspection.

Use only screenshots that have been checked for synthetic data, public catalog records, and absence of credentials or secrets.

## GitHub Social Preview
### Canvas
Target: `1280 x 640` pixels.

### Composition
Use a clean, restrained layout:

- left side: DILIGENT logo and title;
- center or right: one cropped application screenshot, preferably the DILI Agent or Clinical Sessions view;
- short supporting line: `Structured DILI review`;
- secondary line: `Deterministic safeguards + configurable LLMs`.

Do not turn the preview into a dense feature list. It must remain readable at small social-card sizes.

### Visual hierarchy
1. DILIGENT Clinical Copilot
2. Structured DILI review
3. Deterministic safeguards + configurable LLMs
4. Optional small label: Open source, local-first

### Avoid
- `AI diagnosis`;
- `automated causality detection`;
- regulatory logos or visual elements that could imply approval;
- fake clinical dashboards that do not exist in the application;
- fabricated performance statistics;
- patient names, identifiers, images, credentials, private paths, API keys, or other sensitive content.

## Screenshot Set
Prepare a compact public set with one image per major product story rather than many near-duplicates.

### 1. DILI Agent
Show a fictional synthetic case ready for structured review.

Caption:

> Enter clinical context, medication exposure, laboratory findings, and timing before running a structured DILI assessment.

### 2. Clinical Sessions
Show a persisted synthetic report with structured findings visible.

Caption:

> Review persisted reports together with structured evidence, warnings, revisions, and human review state.

### 3. Patient Timeline
Show a synthetic chronology with timeline controls and event inspection.

Caption:

> Inspect longitudinal exposure, symptoms, laboratory events, and chronology without converting uncertain dates into false precision.

### 4. Data Inspection
Show public catalog or knowledge-resource records.

Caption:

> Inspect local data resources, drug records, evidence status, and update state.

### 5. Configurations
Show role-specific model configuration without revealing access keys.

Caption:

> Assign models independently to Clinical, Text extraction, Revision, and Timeline roles using local or supported cloud providers.

## Architecture Graphic
Create one public architecture graphic that explains the application without duplicating internal implementation detail.

Recommended flow:

`Clinical input -> deterministic pre-flight -> structured extraction -> evidence bundle -> drug and knowledge resolution -> optional RAG -> LLM consultation -> faithfulness audit -> persisted report -> human review`

Show these supporting systems below the main flow:

- LiverTox and RxNorm-backed drug knowledge;
- FDA DILIrank 2.0 only when the graphic explicitly describes current development capabilities rather than the published v3.3.0 release;
- local persistence;
- Ollama or supported cloud LLM provider;
- Clinical Sessions and Patient Timeline.

Use distinct visual treatment for deterministic application stages and LLM-assisted stages. The diagram should make it visually obvious that model output does not replace the structured evidence bundle or human review.

## Short Demo Brief
Target a short demonstration, ideally around 60 to 90 seconds, using only a synthetic case.

Suggested sequence:

1. Open Configurations and briefly show local or cloud model roles.
2. Open DILI Agent with a prepared synthetic case.
3. Show pre-flight feedback.
4. Start or show a completed assessment rather than spending most of the recording waiting for inference.
5. Show the structured report and review signals.
6. Open Clinical Sessions.
7. Open the Timeline tab.
8. End on the repository and release information.

The recording should not show real patient information, local private folders, access keys, browser autofill, terminal secrets, or unrelated desktop notifications.

## Demo Captions
Use short overlays only when they explain a real feature:

- `Deterministic pre-flight checks`
- `Structured evidence before narrative`
- `Local Ollama or supported cloud models`
- `Conservative drug identity resolution`
- `Persisted sessions and revisions`
- `Timeline-grounded review`
- `Human review required`

Do not use claims such as `zero hallucinations`, `clinically validated`, `diagnosis in seconds`, or `fully autonomous`.

## Release Versus Development Assets
Keep visual provenance explicit.

For `v3.3.0` promotion:

- use screenshots captured from the release or explicitly verified as representative of that release;
- do not show development-only capabilities and label them as part of `v3.3.0`;
- link to the `v3.3.0` release.

For development previews:

- label the image or post as `development preview`;
- do not provide the preview as evidence that the next release has passed validation;
- keep known release gates documented in the promotion checklist.

## Privacy Review Before Publication
For every image, GIF, or video, check:

- all clinical data are fictional or synthetic;
- no patient identifier is visible;
- no credentials or key fingerprints reveal sensitive information;
- no private local path exposes a user name or confidential directory;
- no browser history, notifications, terminal secrets, or personal desktop content is visible;
- no provider response contains private account information;
- captions describe only capabilities verified in the corresponding release or branch.

## Deliverables
Recommended minimum public visual package:

- one GitHub social preview image;
- four or five curated application screenshots;
- one architecture graphic;
- one short demo GIF or video;
- one square or landscape crop suitable for professional social posts.

Reuse the same verified visual source material across channels rather than creating inconsistent mockups.
