# Promotion Strategy
Last updated: 2026-09-10

## Objective
Promote DILIGENT as an open-source, local-first application for structured Drug-Induced Liver Injury (DILI) review and research without implying diagnostic validation, autonomous clinical decision-making, or regulatory approval.

## Promotion Baseline
- The public repository is `CTCycle/DILIGENT-Clinical-Copilot`.
- The latest published release at this review is `v3.3.0`, published on 2026-09-01 from `main`.
- The published `v3.3.0` release provides a Windows x64 portable executable, MSI installer, and SHA-256 manifest.
- `develop` contains post-release changes and must not be described as equivalent to the published `v3.3.0` release.
- Repository documentation records unresolved revision structured-output and DILI pattern-classification findings that must be resolved and revalidated before a later development state is promoted as a new release.

For promotion performed before the next verified release, link users to `v3.3.0` and describe newer work only as development work.

## Core Positioning
Recommended primary statement:

> DILIGENT is a local-first DILI review application that combines deterministic clinical safeguards, structured evidence, and configurable LLM assistance to make complex hepatotoxicity assessments easier to inspect and review.

The differentiator is not simply that an LLM generates a report. The stronger and better-supported story is that DILIGENT combines model-assisted synthesis with explicit deterministic and auditable structures around the clinical workflow.

### Supported proof points
Promotion may accurately highlight the following implemented behaviors:

- deterministic pre-flight and section-extraction checks before clinical analysis;
- liver-chemistry pattern assessment based on contemporaneous ALT and ALP values normalized to laboratory-specific upper limits of normal;
- explicit handling of competing causes, Hy's Law state, longitudinal exposure and laboratory events, and missing data;
- RUCAM evidence support without inventing a numeric patient RUCAM score when one was not supplied in the patient record;
- conservative separation of patient-specific causality from drug-level LiverTox and FDA DILIrank 2.0 evidence;
- RxNorm and LiverTox-backed drug identity resolution with explicit ambiguous and missing-match states;
- optional retrieval-supported evidence with explicit readiness and fallback behavior;
- configurable local Ollama or supported cloud model providers;
- independent Clinical, Text extraction, Revision, and Timeline model roles;
- persisted clinical sessions, revisions, structured audit artifacts, and patient timelines;
- deterministic faithfulness checks that can prevent a generated or revised report from being treated as successfully finalized when blocking contradictions are detected;
- a packaged Windows desktop distribution as well as source-based operation.

## Claim Boundaries
Use the following language consistently:

- `decision-support application`
- `structured DILI review`
- `reviewable assessment draft`
- `LLM-assisted analysis`
- `deterministic safeguards`
- `human clinical review required`
- `local-first`

Do not describe DILIGENT as:

- an autonomous diagnostic system;
- a validated medical device;
- a replacement for clinician judgment;
- a system that proves a drug caused liver injury;
- a system that automatically calculates a patient RUCAM score when the score is absent from the source record;
- FDA approved, FDA compliant, clinically validated, clinically proven, or production-safe unless a separate documented validation or regulatory basis exists for that exact claim;
- a system that recommends drug rechallenge.

## Target Audiences
Prioritize audiences that can evaluate the implementation rather than broad consumer audiences:

1. DILI and hepatotoxicity researchers.
2. Clinical informatics and medical-AI researchers.
3. Pharmacovigilance and drug-safety practitioners evaluating software workflows.
4. ML and software engineers interested in auditable LLM-assisted clinical applications.
5. Local-first AI and Ollama users interested in domain-specific workflows.

The repository is not positioned as a patient-facing health application.

## Launch Sequence
### Stage 1: Repository readiness
Before external promotion:

1. Keep the public `main` README aligned with the current published release.
2. Use the repository About description and Topics to make the project discoverable without expanding the clinical claims.
3. Add a custom GitHub social preview image using the visual brief in this folder.
4. Keep synthetic-data screenshots prominent and verify they contain no patient information, credentials, or secrets.
5. Verify the release link, portable EXE, MSI, and SHA-256 manifest before linking directly to downloads.

### Stage 2: Targeted soft launch of the published release
Promote `v3.3.0` first to technically relevant audiences. The goal is qualified feedback, not raw star count.

Recommended formats:

- a GitHub release post that explains what a user can actually try;
- a short technical article describing the deterministic plus LLM architecture;
- a focused post to clinical informatics, pharmacovigilance, hepatotoxicity, local-AI, or medical-software communities where self-promotion is permitted;
- a LinkedIn project post aimed at professional and research contacts.

Always disclose that the author is sharing their own project when community rules or context make that relevant.

### Stage 3: Broader technical launch
A Show HN submission or similarly broad launch is better reserved for a state that is easy for an external user to try and has a sufficiently substantial story. Before promoting a newer release broadly:

1. Resolve the documented revision structured-output failure.
2. Resolve the documented DILI pattern-classification inconsistency.
3. Run the repository quality and regression gates applicable to the release.
4. Repeat the portable EXE and MSI host smoke tests against the final tagged commit.
5. Synchronize the release branches according to the repository release process.
6. Publish and verify the new release artifacts and checksums.

## Repository Metadata Recommendations
These settings are maintained in GitHub repository metadata, not in application code.

### About description
Recommended replacement:

> Local-first DILI review and research application combining deterministic clinical safeguards, structured evidence, and configurable LLM assistance.

This is more consistent with the README safety model than wording that implies the software independently detects or manages DILI.

### Topics
The repository currently uses relevant clinical topics. A stronger discoverability set can combine clinical intent and implementation technology while staying specific.

Recommended set:

- `dili`
- `drug-induced-liver-injury`
- `hepatotoxicity`
- `liver-disease`
- `clinical-informatics`
- `clinical-decision-support`
- `pharmacovigilance`
- `toxicology`
- `pharmacology`
- `medical-ai`
- `llm`
- `local-first`
- `ollama`
- `rag`
- `fastapi`
- `angular`
- `tauri`

Do not add generic visibility tags that are unrelated to the project.

### Social preview
Use a 1280 x 640 image where possible. Keep the asset below GitHub's file-size limit for social previews.

Recommended text hierarchy:

1. `DILIGENT Clinical Copilot`
2. `Structured DILI review`
3. `Deterministic safeguards + configurable LLMs`

Use an existing synthetic-data application view as supporting imagery. Do not show real patient data, access keys, private file paths, or unsupported clinical claims.

## Content Angles
Use technical substance rather than generic AI marketing.

### Angle A: Auditable clinical AI
Explain why DILIGENT separates deterministic evidence, LLM synthesis, and human review instead of asking a model for an unstructured answer.

### Angle B: Conservative DILI causality support
Show how the application keeps R-ratio phenotype, competing causes, Hy's Law, RUCAM evidence, drug identity, and drug-level prior knowledge separate rather than collapsing them into one score.

### Angle C: Local-first clinical workflows
Show the Windows desktop package, local persistence, Ollama support, role-specific model configuration, and the option to keep inference local when organizational policy requires it.

### Angle D: Engineering an inspectable agentic revision flow
For a technical audience, describe the bounded revision workflow, persisted plan and tool traces, deterministic patch validation, QA, and the same clinical safety audit used before a revised report can be finalized successfully.

## Channels
Use channels selectively and check each community's current self-promotion rules before posting.

- GitHub repository, Releases, Topics, and social preview.
- Hacker News Show HN when there is a substantial, directly tryable release.
- LinkedIn for a professional project announcement and engineering write-up.
- Relevant Reddit communities only when the post itself provides technical value and complies with local rules.
- Medical-AI, clinical-informatics, pharmacovigilance, hepatology, Python, Angular, Tauri, local-AI, and Ollama communities where project sharing is explicitly allowed.
- Research-oriented articles or posters when claims and evaluation are appropriate to the evidence actually available.

Avoid mass cross-posting the same text, unsolicited promotional comments in unrelated repositories, requests for stars or upvotes, and automated engagement.

## Measurement
Treat stars as a secondary signal. Review the following after each promotion event when the data is available:

- repository visitors and referring sites;
- clones;
- release downloads;
- forks and substantive stars over time;
- technically meaningful issues, pull requests, or discussions generated by the launch;
- installation or startup problems reported by new users;
- conversion from an external article or post to repository visits and release downloads.

Record the date, channel, post URL, release being promoted, and observed outcomes so later promotion is based on evidence rather than repeated posting.

## References
- GitHub repository customization: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository
- GitHub Topics: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/classifying-your-repository-with-topics
- GitHub social preview: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/customizing-your-repositorys-social-media-preview
- GitHub Releases: https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases
- Hacker News guidelines: https://news.ycombinator.com/newsguidelines.html
- Show HN guidelines: https://news.ycombinator.com/showhn.html
