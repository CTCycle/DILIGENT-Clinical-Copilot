# Launch Copy
Last updated: 2026-09-10

## Usage Rule
Unless a post explicitly says that it is discussing development work, the copy below refers to the published `v3.3.0` release. Do not imply that current `develop` has been released or has passed the release gates documented for the next version.

## GitHub About Description
> Local-first DILI review and research application combining deterministic clinical safeguards, structured evidence, and configurable LLM assistance.

## One-line Description
> DILIGENT is an open-source, local-first application for structured Drug-Induced Liver Injury review that combines deterministic clinical checks with configurable LLM-assisted analysis.

## Short Project Description
> DILIGENT Clinical Copilot is an open-source decision-support application for structured Drug-Induced Liver Injury review. It organizes clinical history, medication exposure, laboratory findings, chronology, competing causes, and drug-level evidence into an inspectable workflow, then uses a configured local or cloud model to produce a reviewable assessment draft. Deterministic checks and structured audit artifacts remain separate from model-generated narrative, and human clinical review is required before reuse.

## Technical Project Description
> DILIGENT is a local-first Angular, FastAPI, and Tauri application for structured DILI review. Its clinical workflow separates deterministic extraction and validation, structured evidence construction, drug identity resolution, optional retrieval, LLM consultation, faithfulness auditing, persistence, and human review. It supports local Ollama models as well as supported cloud providers, with separate model roles for clinical reasoning, text extraction, revision, and timeline generation.

## Release Announcement, v3.3.0
### Title
DILIGENT Clinical Copilot v3.3.0: local-first structured DILI review for Windows

### Body
DILIGENT Clinical Copilot v3.3.0 is available as an open-source Windows x64 desktop release, with both a portable executable and MSI installer.

The project is built around a specific constraint: an LLM-generated clinical paragraph should not become the source of truth. DILIGENT keeps the generated narrative alongside structured clinical evidence and review signals covering medication exposure, laboratory patterns, chronology, competing causes, drug matching, Hy's Law, and RUCAM-oriented evidence.

The application can use local Ollama models or supported cloud providers, and it separates Clinical, Text extraction, Revision, and Timeline model roles. Saved sessions can be reviewed, revised, compared, and inspected through a patient timeline and structured audit data.

DILIGENT is a decision-support and research application, not a diagnostic device. Generated content requires qualified human review, and cloud-backed use with real clinical data must follow the user's organizational privacy and data-transfer policies.

The v3.3.0 release includes the portable EXE, MSI installer, and SHA-256 manifest on GitHub Releases.

Feedback is particularly useful on the inspectability of the workflow, local-model usability, installation experience, and the separation between deterministic clinical evidence and LLM-generated text.

Repository: https://github.com/CTCycle/DILIGENT-Clinical-Copilot
Release: https://github.com/CTCycle/DILIGENT-Clinical-Copilot/releases/tag/v3.3.0

## Show HN Draft
### Title
Show HN: DILIGENT, a local-first clinical copilot for structured DILI review

### Body
I built DILIGENT, an open-source desktop application for structured review of Drug-Induced Liver Injury cases.

The main design choice is that the LLM is not treated as the clinical source of truth. The application first builds and preserves structured evidence around exposure chronology, liver-chemistry pattern, competing causes, drug identity, Hy's Law, and RUCAM-oriented evidence. A configured local or cloud model can then help synthesize a reviewable report, while deterministic checks and audit artifacts remain available for inspection.

The application supports local Ollama models and supported cloud providers, with separate model roles for clinical analysis, extraction, revision, and timeline generation. Clinical sessions, revisions, and timelines are persisted locally. The Windows release is available as a portable EXE or MSI.

This is decision-support and research software, not a diagnostic device. It requires human clinical review and is deliberately conservative about causality claims. For example, it does not invent a numeric patient RUCAM score when one is absent from the source record, and drug-level LiverTox or DILIrank information is kept separate from patient-specific causality.

I would be interested in technical feedback on the architecture, auditability, local-model workflow, and whether the deterministic plus LLM separation is understandable to an external reviewer.

Repository: https://github.com/CTCycle/DILIGENT-Clinical-Copilot
Release: https://github.com/CTCycle/DILIGENT-Clinical-Copilot/releases/tag/v3.3.0

## LinkedIn Draft
DILIGENT Clinical Copilot v3.3.0 is now available as an open-source Windows desktop release.

DILIGENT focuses on a narrow problem: making Drug-Induced Liver Injury review more structured and inspectable while using LLMs as assistants rather than sources of clinical truth.

The workflow combines deterministic checks, longitudinal medication and laboratory evidence, liver-pattern assessment, competing-cause review, conservative drug identity resolution, optional retrieval, and configurable local or cloud LLMs. Generated reports remain reviewable drafts, with structured evidence and human review kept explicit throughout the workflow.

The project also supports local Ollama inference, role-specific model configuration, persisted sessions and revisions, patient timelines, and a portable Windows distribution.

It is decision-support and research software, not a diagnostic device, and it is intentionally designed around explicit uncertainty and clinician review.

Project: https://github.com/CTCycle/DILIGENT-Clinical-Copilot

## Technical Community Draft
I am sharing DILIGENT, an open-source local-first application for structured Drug-Induced Liver Injury review, for technical feedback.

The interesting part for this community is the architecture rather than the medical claim. DILIGENT separates deterministic input checks and evidence construction from LLM-assisted synthesis. The persisted workflow keeps chronology, laboratory-pattern evidence, competing causes, drug-resolution state, retrieval provenance, report revisions, and timeline events inspectable instead of relying only on generated prose.

The stack is Angular, FastAPI, local persistence, and Tauri for Windows packaging. Models can run locally through Ollama or through supported cloud providers, with separate roles for clinical analysis, extraction, revision, and timeline generation.

This is not a diagnostic device and the generated assessment is explicitly a draft requiring human review. I am primarily looking for feedback on auditability, failure handling, local-model ergonomics, and the boundary between deterministic application logic and model output.

Repository: https://github.com/CTCycle/DILIGENT-Clinical-Copilot

## Research-oriented Abstract
DILIGENT Clinical Copilot is an open-source, local-first software application for structured Drug-Induced Liver Injury review. The system combines deterministic clinical preprocessing and evidence construction with configurable LLM-assisted extraction, consultation, revision, and timeline generation. The workflow preserves patient-specific chronology, liver-chemistry pattern information, competing-cause states, drug identity resolution, supporting RUCAM evidence, Hy's Law evaluation, and drug-level knowledge as separately inspectable structures. Generated clinical narrative is treated as a reviewable draft and is subject to deterministic faithfulness checks before successful finalization. The application supports local Ollama inference and supported cloud providers and is distributed for Windows as a Tauri desktop package. DILIGENT is intended for decision-support and research workflows and does not replace clinician judgment.

## Short Responses for Common Questions
### Is this a diagnostic tool?
No. DILIGENT is presented as decision-support and research software. Its generated report is a draft and requires qualified clinical review.

### Does it calculate RUCAM automatically?
DILIGENT can structure RUCAM-oriented evidence, but it does not synthesize a numeric patient RUCAM score when one was not supplied in the current patient's record.

### Can it run locally?
Yes. The application supports local Ollama models. Cloud providers are also supported where configured.

### Does it use LiverTox?
Yes. LiverTox evidence is used as drug-level knowledge after conservative identity resolution. It is not treated as patient-specific proof of causality.

### Does it use FDA DILIrank 2.0?
The current development documentation includes FDA DILIrank 2.0 as structured drug-level hepatotoxicity knowledge. Do not claim that DILIrank 2.0 is part of the published `v3.3.0` binary unless release-specific verification confirms it.

### Can I use patient data with a cloud model?
Only when the relevant organization explicitly permits that data transfer. DILIGENT's documentation advises against entering real patient information into a cloud-backed workflow unless the transfer is authorized.

## Copy to Avoid
Do not publish statements such as:

- `DILIGENT diagnoses DILI.`
- `DILIGENT determines which drug caused liver injury.`
- `DILIGENT replaces RUCAM or expert adjudication.`
- `DILIGENT is clinically validated.`
- `DILIGENT is FDA approved.`
- `DILIGENT safely manages patient treatment.`
- `DILIGENT recommends rechallenge.`
- `The AI eliminates hallucinations.`

These claims are not supported by the repository documentation and conflict with the project's stated safety boundaries.
