# LLM-assisted revision end-to-end validation

Execution window: 2026-09-16, Europe/Rome
Repository: `DILIGENT Clinical Copilot`
Branch: `develop`
Browser: Chrome through the Codex in-app browser tooling, normal desktop viewport
Frontend used for the primary run: `http://127.0.0.1:9848/clinical-sessions`
Backend: `http://127.0.0.1:7690`
Test data: synthetic clinical sessions only; no real patient identifiers

## Scope and model lane

This validation covered only LLM-powered revision of already persisted clinical sessions. Manual editing, session creation, initial DILI analysis, RAG, timeline generation, configuration editing, image upload, and general database auditing were not tested.

The visible Revision model was `deepseek-v4-flash`, mapped by the application to the DeepSeek Flash 4.1 semantic model (`deepseek-v4.1-flash`). The persisted task trace recorded provider `opencode_go` and model `deepseek-v4-flash` for every tested model step. No provider or model fallback was used.

The browser workflow was the source of truth. Read-only database queries, backend logs, and source inspection were used only to corroborate rendered state or diagnose observed failures. Revision starts and instructions were entered through the UI.

## Existing sessions and baseline

Two meaningful persisted sessions were used:

| Session | Existing content used as context | Baseline evidence |
|---|---|---|
| Session 18, `Synthetic Case C`, Version 1 | Capmatinib exposure, ALT 166, AST 73, ALP 194, GGT 322, bilirubin 1.1, indeterminate pattern, clinical report | Source version 21; original report length 9,412 characters |
| Session 19, `Synthetic Case C`, Version 2 | Same clinical case with a distinct persisted report and a manual QA marker | Source version 26; original report length 9,123 characters |

The UI showed both sessions as existing `Successful` entries before revision. The original Session 19 Preview continued to show only its original `Manual QA marker: Session 19 persistence check 2026-09-16` after all revision runs; it did not show the instructed sentence.

## Browser evidence

CUA screenshots were emitted and visually inspected inline in the task trace at these checkpoints:

1. Clean Session 18 Revision panel with the configured `deepseek-v4-flash` model and blank instruction.
2. Session 19 with the exact instruction captured in the Revision textbox and the disabled running-state controls.
3. Completed Session 19 Revision panel after the live-status fix, showing `Revision completed with QA issues.`, the `deepseek-v4-flash` model, and the new Session 22 at the top of the session list.
4. Session 22 after reload, showing the appended sentence in the Clinical Report.
5. Session 19 Preview after returning from Session 22, showing the original report without the appended sentence.

The final browser console check returned no `error` or `warn` entries. The backend log used for corroboration was [DILIGENT_20260916_182107_879344_17396.log](<G:/Projects/Repositories/Active projects/DILIGENT Clinical Copilot/app/resources/logs/DILIGENT_20260916_182107_879344_17396.log>).

## Test cases and results

| Case | User interaction | Expected behavior | Observed behavior | Result |
|---|---|---|---|---|
| A1 | Open existing Session 18 and select Revision with no instruction | Existing content is loaded; model is visible; instruction is blank; Start is available | Session 18 opened with the capmatinib clinical content, blank Revision instruction, visible `deepseek-v4-flash`, and enabled Start revision | PASS |
| A2 | Start the one-shot revision for Session 18 without additional instructions | Model reviews the existing session, produces a meaningful draft, and does not overwrite the original | UI entered the running state. Run 11 completed; the draft retained the patient, visit date, capmatinib, laboratory values, indeterminate pattern, warnings, and clinical-review language. The original Session 18 remained Version 1 | PASS for contextual draft generation and original preservation; QA acceptance FAIL because the safety QA gate found two blocking issues |
| A3 | Navigate away from Session 18, return, switch Revision, and reload | The saved revision artifact remains accessible and the original remains unchanged | The generated draft, audit state, and `Revision completed with QA issues.` status reappeared after navigation and reload; Session 18 remained the original session row | PASS for persistence of the review draft |
| B1 | Open existing Session 19 and enter the exact instruction: `Append exactly this sentence to the revised report: Reviewer instruction check: human clinical review is required before reuse. Preserve all existing clinical facts and sections; do not remove or rewrite unrelated content.` | The instruction is captured, transmitted, and materially changes the draft | The exact text was visible in the textbox and persisted in the revision context with `truncated: false` | PASS |
| B2 | Start the instructed revision before the append guard | The exact sentence appears in the revised report | Run 12 completed but produced no patch and omitted the sentence; the UI surfaced QA failed | FAIL; corrected below |
| B3 | Repeat after prompt-only clarification | Prompt clarification alone should remove the omission | Run 13 still produced no patch and omitted the sentence | FAIL; prompt-only remediation was insufficient |
| B4 | Repeat after deterministic append guard | Exact sentence is appended without changing unrelated canonical content | Run 14 produced a validated append patch at offset 9,123 and rendered the sentence, but QA found two unrelated safety blockers | PASS for instruction compliance; QA acceptance FAIL |
| B5 | Repeat through the UI after the append guard and live-status fix | The exact instruction is followed, a valid revision may be persisted, and the terminal banner reflects QA | Run 15 completed successfully: target revision 39 became Session 22, the draft ended with the exact sentence, QA had zero blocking issues, and the new session appeared in the UI after reload | PASS |
| B6 | Return to Session 19, start the same exact instruction from a clean loaded view after the stale-load guard | Starting a new revision must clear old artifacts and retain the new running state | Run 16 settled at 0 steps/0 artifacts with Start disabled, Cancel visible, the exact instruction retained, and the working banner; the late persisted-draft load did not replace this state | PASS |
| B7 | Observe Run 16 terminal state and compare the original | QA outcome is visible, the instruction is reflected, and the original is unchanged | The live banner read `Revision completed with QA issues.`, the exact sentence was rendered, 5 steps and 5 artifacts were shown, and Session 19 Preview still lacked the sentence | PASS for state handling, instruction compliance, and original preservation; QA acceptance FAIL because the model draft had four safety/data blockers |
| C1 | Start a revision, click Cancel, and inspect the resulting state | Cancellation terminates progress, leaves a truthful status, and permits a later retry | Run 10 was cancelled through the UI; the UI showed `Cancellation requested.`, the backend log recorded `CancelledError()`, and no stale working message remained | PASS |
| C2 | Switch sessions after cancellation and after completed revisions | Revision instruction, audit, status, and draft content do not leak between sessions | State reset and generation guards restored a clean panel when switching; Session 18, Session 19, and Session 22 displayed their corresponding content | PASS |

## Persistence and versioning evidence

The following read-only persistence observations corroborated the UI:

| Run | Source | Target version | Target session | Final status | Key evidence |
|---:|---|---:|---:|---|---|
| 11 | Session 18 / version 21 | 35 | none | `qa_failed` | No instruction; draft length 9,412; 2 QA blockers; persisted draft reloaded in the UI |
| 12 | Session 19 / version 26 | 36 | none | `qa_failed` | Exact instruction stored; 0 patches; requested sentence absent |
| 13 | Session 19 / version 26 | 37 | none | `qa_failed` | Prompt clarification did not correct the omission |
| 14 | Session 19 / version 26 | 38 | none | `qa_failed` | Exact append patch present; 2 QA blockers |
| 15 | Session 19 / version 26 | 39 | Session 22 | `llm_qa_passed` | Exact append patch present; 0 QA blockers; new Session 22 persisted and reopened after reload |
| 16 | Session 19 / version 26 | 40 | none | `qa_failed` | Exact append patch present; 4 model-safety/data blockers; UI correctly displayed QA issues |

Run 15 established the accepted revision lineage: Session 22 was created from root Session 19 and source version 26. Its Version 6 row remained separate from the original Session 19 Version 2. Failed QA runs remained draft revision shells with no target session, so they did not overwrite or masquerade as successful sessions. Revision version numbers remained ordered across the root (failed draft versions 3, 4, 5, then accepted version 6, then failed draft version 7).

The Run 15 context artifact recorded the full exact instruction with `truncated: false`. Its draft artifact contained a single append patch at the canonical report end, and its QA artifact contained zero blocking issues. The Run 16 context and draft independently reproduced the same instruction and append patch, demonstrating that the behavior was not limited to a stale displayed artifact.

## Initial defects, root causes, and surgical remediation

### 1. Native revision tools were rejected by the configured provider

- Initial result: Session 21 one-shot revision failed in the UI at the first native tool call with HTTP 400 from the configured OpenCode Go endpoint. The audit showed a failed task and the error `Cloud provider returned HTTP 400`.
- Root cause: no-argument `ToolDefinition` instances serialized the default `{}` as the tool parameters schema. The native provider required an object schema.
- Remediation: commit `dc2a43a4` added one shared empty object schema with `type: object`, empty `properties`, and `additionalProperties: false`, and applied it only to the three no-argument revision context tools in [revision_tools.py](<G:/Projects/Repositories/Active projects/DILIGENT Clinical Copilot/app/server/services/inspection/revision_tools.py>).
- Retest: subsequent native tool calls for the revision workflow returned HTTP 200, and the no-instruction Run 11 completed through draft and QA persistence rather than failing at tool invocation.
- Impact: high for the configured revision lane; every affected revision was blocked before the model could review the session.

### 2. Cancellation status and revision state became stale

- Initial result: after cancelling an in-flight revision, the UI re-enabled Start but a late poll restored `Revision agent is working...`. Opening another session inherited cancellation text and audit state.
- Root cause: the in-flight poll updated state after cancellation, and `openSession` did not reset revision-only signals when the new session had no persisted revision.
- Remediation: commit `a829bc50` marks the poll cancelled and updates the status before awaiting cancellation, ignores late poll results, and resets instruction/status/job/version/steps/artifacts/draft state when opening or clearing a session.
- Retest: Run 10 showed `Cancellation requested.` with Start enabled and no stale audit; switching sessions produced clean revision state.
- Impact: medium/high usability and trust issue; users could see another session’s status or believe a cancelled model was still running.

### 3. Cancellation did not interrupt an in-flight cloud request

- Initial result: after Run 8 was cancelled in the UI, a retry returned `Another operation is already running. Please wait and retry.` because the long-running provider request still held the operation slot.
- Root cause: non-streaming OpenAI-compatible transport awaited the HTTP retry task without polling the cancellation predicate.
- Remediation: commit `27c2b940` runs the request as a cancellable task, checks `cancel_check` at short intervals, and cancels/awaits the HTTP task on cancellation. It also adds a focused unit test.
- Retest: Run 10 produced an HTTP receive `CancelledError()` and the backend marked the job cancelled; the UI retained the truthful cancellation status and a later start was not blocked by the abandoned request.
- Impact: high for recovery from provider latency; without the fix a user could be unable to retry for the transport timeout window.

### 4. The final editor response was truncated while echoing the full report

- Initial result: Session 18 Run 7 reached the editor but all four structured parse attempts failed on a 14,640-character response with `JSONDecodeError: Unterminated string`. No generated draft was available.
- Root cause: the editor contract required a full report echo even though validated deterministic patches were authoritative; the provider response was cut in the middle of a large JSON string.
- Remediation: commit `6ac94294` changed the editor contract to allow an empty `revised_report_text`, instructed the model to return compact patch-only output, and makes the runner derive the persisted report from validated patches when model text is empty.
- Retest: Run 11 completed the editor, produced a meaningful draft artifact, and passed structured parsing; Runs 14-16 also produced valid draft artifacts.
- Impact: high; the user could not obtain any review draft despite successful earlier revision steps.

### 5. Planner limits were too small for real session context

- Initial result: the first planner-cap attempt caused Run 9 to fail parsing a 15,179-character planner response with an unterminated JSON string.
- Root cause: the initial remediation imposed arbitrary low field limits on a planner response carrying substantial clinical context.
- Remediation: commit `4fad29c5` kept limits generous (`objective` 4,000, `instruction_profile` 4,000, `stop_criteria` 2,000, and similarly bounded but ample fields), limited only the existing maximum of eight tasks, and explicitly instructed the planner to preserve materially distinct issues and needed execution detail rather than omit content solely for a character cap.
- Retest: Run 11 planner requests and repair requests returned HTTP 200; the planner completed and the run progressed to draft/QA. No arbitrary low response cap was retained.
- Impact: high for meaningful sessions; the planner could fail before any review work began.

### 6. Explicit append instructions were captured but not followed

- Initial result: Run 12 stored the exact instruction and planned an append task, but the model returned no patch and claimed the append offset was ambiguous because of the trailing QA marker. Run 13 repeated the omission after a prompt-only clarification.
- Root cause: the model/editor could see the canonical report and the instruction but refused a mechanically determinable append operation; this was not a frontend capture or serialization failure.
- Remediation: commit `193118db` clarified the exact append contract in the editor prompt. When that did not resolve the reproducible omission, commit `e8067d86` added a narrow deterministic guard for the explicit `Append exactly this sentence to the revised report:` instruction. It appends only the requested sentence at the end of the canonical report, preserves all other characters, and records `user_revision_instruction` as evidence.
- Retest: Runs 14, 15, and 16 all stored and rendered the exact sentence. Run 15 passed QA and created Session 22; Run 16 also confirmed the result after a fresh UI start.
- Impact: high for instruction-following; without the guard, the user’s explicit revision request was silently omitted.

### 7. Live terminal status hid a persisted QA failure

- Initial result: after a completed QA-failed run, the live poll banner displayed only `completed` while the same audit showed `Quality review — QA failed`. Reloading later showed the more accurate `Revision completed with QA issues.` text.
- Root cause: the live poll rendered the generic job status and ignored the returned `result.revision_status`.
- Remediation: commit `b01383cc` maps `result.revision_status === 'qa_failed'` to the existing QA-aware message while preserving the normal running and generic failure messages.
- Retest: Run 16’s terminal UI displayed `Revision completed with QA issues.`, Start was re-enabled, the generated draft and audit remained visible, and browser console errors/warnings were empty.
- Impact: medium/high clinical trust issue; a QA-gated draft could appear fully completed until the user navigated away.

### 8. A late persisted-draft load overwrote a newly started run

- Initial result: after a clean Session 19 start, the backend continued a new run while a late asynchronous persisted-revision load restored the previous draft and re-enabled Start. This created a visible mismatch between the running backend and the UI and could invite a duplicate start.
- Root cause: `loadPersistedRevision` checked only the selected session, not whether its asynchronous request belonged to the current revision state.
- Remediation: commit `4a69fc77` adds a monotonic revision-load generation. Opening/clearing a session or starting a revision invalidates earlier loads, and checks are performed after each awaited steps/artifacts request before any stale state is written.
- Retest: Run 16 was started immediately after selecting Session 19. Once settled, the UI showed 0 steps/0 artifacts, Start disabled, Cancel visible, the exact instruction, and the working banner; the previous draft did not replace the new run. The same run later completed with the QA-aware terminal message and correct draft.
- Impact: high for state integrity and duplicate prevention in a slow network/provider environment.

## Focused automated/supporting checks

After the append guard, the focused backend suite completed with `31 passed, 2 warnings`:

```text
tests/unit/test_deepseek_v41_compatibility.py
tests/unit/test_revision_agent_skeleton.py
```

The frontend Angular development watcher rebuilt successfully after the two client-side fixes and reported `Application bundle generation complete`. A separate production `RebuildFrontend` launcher attempt earlier in the session terminated with Windows exit code `-1073741819`; this was an environment/toolchain build limitation, while the browser validation used the successfully rebuilt dev bundle.

## Unresolved defects and limitations

1. The no-instruction Run 11 generated and persisted a meaningful review draft, but QA correctly rejected it because it claimed review with no validated edits and returned non-schema-conformant issue metadata. No accepted session was created. This remains a workflow acceptance failure, not a parser failure after the fixes.
2. Provider/model output was variable across repeated exact instruction runs. Run 15 passed QA and created Session 22, while Run 16 followed the instruction but was QA-rejected for four safety/data consistency blockers. The application surfaced the rejection and did not create a misleading successful session, but repeated acceptance is not deterministic on this provider lane.
3. The OpenCode Go lane showed substantial latency, including multi-minute requests and one HTTP 500 retry during Run 15. The workflow recovered, but latency remains an operational risk for users.
4. Production frontend rebuild was not available in this environment because of the launcher/toolchain crash described above. Dev-bundle browser evidence is valid for this session; release packaging remains unverified.

No unrelated defects were remediated. Manual editing and other application workflows were intentionally left untested.

## Scoped reliability conclusion

The tested LLM-powered revision subset using DeepSeek Flash 4.1 through `opencode_go` is **partially reliable**:

- Reliable after remediation for opening existing sessions, capturing and transmitting explicit instructions, executing the configured revision lane, preserving original sessions, retaining auditable draft artifacts, handling cancellation, preventing stale session state, and surfacing QA failure truthfully.
- Demonstrated end-to-end accepted revision for the explicit instruction: Run 15 created and persisted Session 22, and the result remained accessible after navigation and reload.
- Not fully reliable for the no-instruction case or for repeated provider runs to pass the clinical QA gate; Run 11 and Run 16 were correctly blocked from becoming accepted sessions.

This conclusion applies only to LLM-powered revision of existing sessions on the tested DeepSeek Flash 4.1/OpenCode Go lane. It does not generalize to the rest of the application.
