# DILIGENT Selective E2E Evaluation

Date: 2026-04-01
Application: `http://127.0.0.1:7861`
Configured target model: `gpt-4.1-mini`
Evaluation scope: 5 random DILI-reference cases from `C:\Users\Thomas V\Desktop\DILI_analysis\DILI`

## 1. Summary of tested cases

Tested case PDFs:

1. `244623867.pdf`
2. `241033960.pdf`
3. `370833380.pdf`
4. `297083444.pdf`
5. `237723396.pdf`

Execution summary:

- 1/5 cases produced a full application report.
- 4/5 cases failed before report generation with extraction-stage errors.
- Only the first case actually executed with `cloud=True` and `gpt-4.1-mini`.
- Cases 2-5 silently ran with `cloud=False`, `parsing_provider=ollama`, `clinical_provider=ollama`, and then failed.

## 2. Per-case comparison

### Case 244623867

Reference extraction:

- Anamnesis: 80-year-old patient with pyoderma gangrenosum, AKI on CKD, atrial fibrillation, cholestatic enzyme abnormalities during antibiotic therapy, and POCUS positive for cholelithiasis.
- Therapy examined in conclusions: `Iberogast`, `Co-amoxi / amoxicillin-clavulanate`, `Piperacillin-tazobactam`.
- Ground truth conclusion:
  - overall role of pharmacotherapy in the initial clinical picture is `poco probabile`
  - some drugs may have contributed to worsening bilirubin/ALP, `especially Co-amoxi and Piperacillina-tazobactam`
  - `Iberogast` should be suspended until acute episode resolves

Observed application output:

- Full report generated.
- KB behavior:
  - `Piperacillin-tazobactam` matched
  - `Co-amoxicillin / amoxicillin-clavulanate` ambiguous
  - `Iberogast` no KB match
  - spurious token `Case` treated as a drug mention
- Final synthesis stated:
  - probable `co-amoxicillin` DILI superimposed on baseline cholestasis
  - `piperacillin-tazobactam` lower suspicion
  - `Iberogast` unlikely implicated and continuation acceptable

Comparison:

- Mismatch: physician conclusion says the initial drug role is low-probability with possible worsening contribution; app upgrades co-amoxicillin to a stronger causal interpretation.
- Mismatch: physician explicitly recommends suspending Iberogast; app recommends continuing it.
- Mismatch: app could not resolve the key combination product in KB, weakening the main causality step.
- Mismatch: app introduced a false drug token (`Case`), indicating extraction noise.

### Case 241033960

Reference extraction:

- Anamnesis: 75-year-old woman with jaundice under investigation; high bilirubin and ALP; no hepatobiliary obstruction on CT.
- Therapy examined in conclusions: `Co-amoxi`, `Ceftriaxone`, `Quetiapina`.
- Ground truth conclusion:
  - `Co-amoxi`: possible
  - `Ceftriaxone`: possible, but less likely than Co-amoxi
  - `Quetiapina`: improbable

Observed application output:

- UI output: `[ERROR] Failed to extract drugs from anamnesis via LLM`
- No clinical report generated.
- Backend runtime for this request resolved to:
  - `cloud=False`
  - `parsing_provider=ollama`
  - `clinical_provider=ollama`
  - `cloud_model=gpt-5.4`

Comparison:

- No usable medical comparison possible because the app failed before causality output.
- Failure is still product-critical because it blocks evaluation of a straightforward physician-labeled case.

### Case 370833380

Reference extraction:

- Anamnesis: 74-year-old woman with RUQ pain, dark urine, chills, cholestatic/cytolytic hepatopathy.
- Therapy examined in conclusions: `Fluvastatina`, `Ezetimibe`, `Metformina`.
- Ground truth conclusion:
  - `Fluvastatina`: temporally supportive causal role
  - `Ezetimibe`: temporally supportive causal role
  - `Metformina`: temporal correlation not supportive, not fully excluded

Observed application output:

- UI output: `[ERROR] Failed to extract drugs from anamnesis via LLM`
- No clinical report generated.
- Backend runtime again resolved to local/Ollama instead of the requested cloud model.

Comparison:

- No usable medical comparison possible because the app failed before report generation.

### Case 297083444

Reference extraction:

- Anamnesis: recurrent ALT elevation and pyrexia on `Encorafenib + Binimetinib`; question about switch to `Dabrafenib + Trametinib`.
- Therapy examined in conclusions: current `E+B` regimen and proposed `D+T`.
- Ground truth conclusion:
  - switch to `D+T` is pharmacologically acceptable
  - no toxicity-based contraindication to switch
  - `D+T` has a better hepatic safety profile but higher pyrexia risk
  - close liver monitoring remains necessary

Observed application output:

- UI output: `[ERROR] Failed to extract drugs via LLM`
- No clinical report generated.
- App also raised a modal about missing ALT/ALP labs before proceeding.
- Backend runtime again resolved to local/Ollama instead of the requested cloud model.

Comparison:

- No usable medical comparison possible because the app failed before a clinical answer.

### Case 237723396

Reference extraction:

- Anamnesis: 60-year-old woman with persistent fever, rash, and mixed liver injury; possible DILI.
- Therapy examined in conclusions: `Bactrim`, `Trastuzumab deruxtecan`, `Pregabalin`.
- Ground truth conclusion:
  - `Bactrim`: possible
  - `Trastuzumab deruxtecan`: possible, but less likely than Bactrim
  - `Pregabalin`: improbable

Observed application output:

- UI output: `[ERROR] Failed to extract drugs from anamnesis via LLM`
- No clinical report generated.
- Backend runtime again resolved to local/Ollama instead of the requested cloud model.

Comparison:

- No usable medical comparison possible because the app failed before report generation.

## 3. Identified discrepancies

1. Runtime-selection regression after the first case

- Case 1 used `cloud=True` with `gpt-4.1-mini`.
- Cases 2-5 resolved to `cloud=False` with `ollama` parsing/clinical models, despite the server config endpoint still reporting `use_cloud_services=true` and `cloud_model=gpt-4.1-mini`.
- This is the most severe issue because it invalidates later runs and makes the UI behavior inconsistent with the configured model.

2. LLM extraction hard failures block report generation

- 4/5 cases terminated with:
  - `Failed to extract drugs from anamnesis via LLM`, or
  - `Failed to extract drugs via LLM`
- This is a product-blocking failure mode, not a minor quality defect.

3. KB matching is weak for combination and non-standard drug names

- In the only successful case:
  - `Co-amoxicillin / amoxicillin-clavulanate` remained ambiguous
  - `Iberogast` had no KB match
  - non-drug token `Case` was treated as a candidate drug
- This undermines causality ranking even when the job completes.

4. The generated narrative overstates causality relative to physician ground truth

- Case 244623867:
  - ground truth: overall drug role initially low-probability; worsening contribution possible
  - app: frames co-amoxicillin as probable DILI
- The app is not preserving the narrower physician uncertainty.

5. Recommendation drift against source-ground-truth conclusions

- Case 244623867:
  - physician: suspend Iberogast
  - app: Iberogast continuation acceptable

## 4. Issue categories

- Runtime/configuration: silent fallback from cloud GPT flow to local/Ollama flow
- Prompting/orchestration: anamnesis drug extraction stage brittle enough to crash the workflow
- KB matching: ambiguous handling of combination products and herbal products
- Context hygiene: non-drug text (`Case`) becomes a drug candidate
- Clinical reasoning: stronger-than-justified causality wording and contradictory recommendation

## 5. Frequency and patterns

- Runtime mismatch: 4/5 cases
- Extraction-stage fatal failure: 4/5 cases
- Successful end-to-end report generation: 1/5 cases
- KB matching defects among successful runs: 1/1 successful case
- Ground-truth contradiction in recommendation among successful runs: 1/1 successful case

Observed pattern:

- The product is not failing randomly case-by-case.
- After one successful run, subsequent jobs consistently route to the wrong runtime and then fail in the same extraction stage.
- Even when the flow completes, drug normalization and causality narration remain unreliable for real-world DILI inputs involving combination products and herbal agents.

## 6. Prioritized concrete findings

P0. Fix runtime resolution for repeated clinical jobs.

- The app was configured for `gpt-4.1-mini`, but jobs 2-5 executed with `cloud=False` and `ollama` providers.
- This invalidates evaluation reproducibility and causes hard failures when Ollama is unavailable.

P0. Make anamnesis drug extraction non-fatal.

- If extraction fails, the workflow should degrade gracefully and continue with the explicit therapy list rather than aborting the entire report.

P1. Improve KB normalization for combination products and herbal agents.

- `Co-amoxi / amoxicillin-clavulanate` should normalize deterministically.
- `Iberogast` and similar branded herbal products need either direct mapping or structured decomposition.

P1. Add strict filtering for non-drug tokens before KB lookup.

- Terms like `Case` must never enter the drug candidate set.

P1. Constrain final synthesis to source-ground-truth uncertainty.

- The model should avoid upgrading `possible contributor` into `probable DILI` when the source context points to mixed competing causes.

P1. Add contradiction checks for recommendations.

- If the physician conclusion recommends suspending a drug, the generated recommendation should not say continuation is acceptable without explicit contrary evidence from the case.

## 7. Key evidence points

- Backend log evidence showed:
  - job `5901d30c`: `cloud=True`, `gpt-4.1-mini`, completed successfully
  - jobs `7d599619`, `8fce8006`, `45daf5ea`, `ae0ea2f7`: `cloud=False`, `ollama` parsing/clinical, failed
- The server-side `/model-config` endpoint still reported:
  - `use_cloud_services=true`
  - `cloud_model=gpt-4.1-mini`
- Therefore the later failures are consistent with a request-time runtime-resolution bug, not simply a missing saved configuration.
