# Promotion Checklist
Last updated: 2026-09-10

## Purpose
Use this checklist before promoting DILIGENT externally. It separates repository presentation work from release validation so promotional material cannot silently outrun the software state.

## A. Repository Presentation
Before a public campaign:

- [ ] Confirm the repository is public and the intended release is accessible without authentication.
- [ ] Confirm the `main` README describes the published release rather than unreleased `develop` behavior.
- [ ] Replace the GitHub About description with the conservative description in `promotion_strategy.md`.
- [ ] Review and apply the recommended GitHub Topics in `promotion_strategy.md`.
- [ ] Add a custom social preview image following `visual_asset_brief.md`.
- [ ] Confirm screenshots contain only synthetic clinical data and public catalog records.
- [ ] Confirm no access key, credential, private file path, personal desktop information, or real patient identifier appears in public assets.
- [ ] Confirm the license remains visible from the repository landing page.
- [ ] Consider adding `CITATION.cff` before research-oriented outreach, but only after author, title, release version, and preferred citation metadata have been explicitly reviewed.
- [ ] Do not add contribution, security, citation, or governance files merely as decoration. Add them only when the project owner is prepared to maintain the stated process.

## B. Published v3.3.0 Promotion Gate
The existing `v3.3.0` release may be promoted only as the published `v3.3.0` state.

Before linking directly to it:

- [ ] Confirm `v3.3.0` remains the intended release to promote.
- [ ] Confirm the GitHub Release is not a draft or prerelease.
- [ ] Confirm the portable Windows x64 executable is attached.
- [ ] Confirm the Windows x64 MSI is attached.
- [ ] Confirm the SHA-256 manifest is attached.
- [ ] Use release-specific screenshots when making claims about `v3.3.0`.
- [ ] Avoid claiming development-only behavior as a `v3.3.0` capability.
- [ ] Include the decision-support and human-review boundary in clinical or research-facing posts.

## C. Next-release Promotion Gate
Do not broadly promote a post-v3.3.0 development state as a finished release until all applicable release gates pass.

At minimum:

- [ ] Resolve the documented revision structured-output failure.
- [ ] Resolve the documented DILI pattern-classification inconsistency.
- [ ] Run backend compilation and repository quality checks required by CI.
- [ ] Run Alembic migration and metadata-drift checks.
- [ ] Run Ruff.
- [ ] Run Pyright.
- [ ] Run backend unit tests.
- [ ] Run persistence contract tests for the supported persistence backends used by CI.
- [ ] Run Angular/Vitest tests with the repository CI command.
- [ ] Build the Angular frontend.
- [ ] Run the documented Windows regression slice where applicable.
- [ ] Build release artifacts from the final intended release commit.
- [ ] Run the portable EXE host smoke test.
- [ ] Run the MSI host smoke test appropriate to the release process.
- [ ] Verify release artifact checksums.
- [ ] Synchronize `develop` and `main` according to the release process before tagging.
- [ ] Confirm the final tag points to the intended release commit.
- [ ] Confirm published release metadata and downloaded artifact hashes after publication.

Do not infer a successful release from source tests alone. Packaged-artifact validation remains a separate gate.

## D. Copy Review
For every post:

- [ ] Identify whether the post describes `v3.3.0` or a development preview.
- [ ] Use `decision-support`, `structured review`, or equivalent conservative terminology.
- [ ] Describe model output as a draft or assisted synthesis, not an autonomous diagnosis.
- [ ] Keep patient-specific causality separate from drug-level LiverTox or DILIrank knowledge.
- [ ] Do not claim automatic patient RUCAM scoring when no score is supplied in the patient record.
- [ ] Do not claim regulatory approval, clinical validation, diagnostic performance, accuracy, sensitivity, specificity, or improved patient outcomes without direct evidence supporting that exact claim.
- [ ] Do not claim hallucinations are eliminated.
- [ ] Do not imply the software recommends drug rechallenge.
- [ ] Include human-review and privacy boundaries where the audience could reasonably interpret the project as ready for clinical use.

## E. Community Posting Hygiene
Before posting outside GitHub:

- [ ] Read the destination's current self-promotion and project-sharing rules on the day of posting.
- [ ] Disclose that this is the author's own project when appropriate.
- [ ] Make the post useful without requiring a star, follow, or upvote.
- [ ] Never solicit coordinated upvotes or stars.
- [ ] Do not paste promotional comments into unrelated GitHub issues, pull requests, or discussions.
- [ ] Do not mass-post the same copy across unrelated communities.
- [ ] Adapt the technical angle to the community.
- [ ] Link to the repository or release only when it directly supports the post.
- [ ] Be prepared to answer installation, architecture, licensing, privacy, and clinical-boundary questions.

For Hacker News specifically, use Show HN only when external users can meaningfully try the project and the submission represents a substantial project or release rather than a trivial update.

## F. Recommended First Campaign
For the current repository state, use a controlled sequence:

1. Apply GitHub About description, Topics, and social preview updates.
2. Verify the `v3.3.0` release and its public assets.
3. Publish one technical project article or detailed professional post explaining the deterministic plus LLM architecture.
4. Share the project with one or two highly relevant communities whose rules permit project posts.
5. Collect installation and architecture feedback before broadening distribution.
6. Reserve a broader Show HN-style launch for a later release after the documented development release gates are cleared.

## G. Measurement Log
For every external promotion event, record:

- date;
- promoted release or branch;
- channel;
- post URL;
- positioning angle;
- repository visitors and referring sites when available;
- clones when available;
- release download changes;
- stars and forks as secondary indicators;
- substantive issues, pull requests, or discussions generated;
- installation or runtime failures reported by new users;
- corrective actions required after feedback.

Review results after roughly 7 days and again after 30 days where traffic data remains available. Avoid reposting solely because vanity metrics were low.

## H. Stop Conditions
Pause active promotion if any of the following occurs:

- a promoted release is found to have a clinically material calculation or evidence-grounding defect;
- a release artifact or checksum cannot be verified;
- a public screenshot contains sensitive or private information;
- a promotional statement materially overstates the documented capabilities;
- new-user feedback reveals a reproducible installation blocker that prevents the advertised workflow;
- the linked release is withdrawn, superseded for safety reasons, or otherwise no longer the intended public build.

Correct the underlying problem and public documentation before resuming promotion.

## I. Manual GitHub Metadata Actions
The following changes are recommendations in this branch and are not application-code changes:

1. Update the repository About description using the value in `promotion_strategy.md`.
2. Apply the recommended Topics after reviewing them for current relevance.
3. Upload the reviewed social preview image in repository settings.
4. Enable GitHub Discussions only if the project owner intends to actively moderate and use it as a support or feedback channel.
5. Add a project homepage or GitHub Pages site only when there is a maintained destination worth linking from the repository metadata.

## J. Repository-specific Validation Commands
These commands are documented by the current repository and are relevant when a later software release is promoted. They are not required solely because this promotion documentation changes.

Frontend tests:

```bash
cd app/client
npm run test -- --no-watch
```

Frontend build:

```bash
cd app/client
npm run build
```

Backend checks used by CI are run from `app/server` after the locked environment is installed:

```bash
python -m compileall -q . ../tests
python -m alembic -c alembic.ini upgrade head
python -m alembic -c alembic.ini current --check-heads
python -m alembic -c alembic.ini check
python -m ruff check . ../tests
python -m pyright .
python -m pytest ../tests/unit -q
```

The documented Windows focused regression runner is:

```cmd
app\tests\run_tests.bat modelconfig
```

The full model-configuration/app-flow regression variant is:

```cmd
app\tests\run_tests.bat modelconfigfull
```

Use the release-specific desktop validation process in `assets/docs/runtime/desktop_release.md` for packaged artifact validation.
