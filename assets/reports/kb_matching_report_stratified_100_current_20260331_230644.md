# KB Matching Evaluation (Stratified 100, Current)

## Files
- Baseline JSON: `G:\Projects\Repositories\Active projects\DILIGENT Clinical Copilot\assets\reports\kb_matching_report_stratified_100.json`
- Current JSON: `G:\Projects\Repositories\Active projects\DILIGENT Clinical Copilot\assets\reports\kb_matching_report_stratified_100_current_20260331_230644.json`

## Real Alias Recall
- with_excerpt: baseline 96.00% -> current 92.00% (delta -4.00 pp)
- without_excerpt: baseline 2.00% -> current 0.00% (delta -2.00 pp)

## Accepted-Match Precision by Stratum
- with_excerpt: 119/120 (99.17%)
- without_excerpt: 1/6 (16.67%)

## Canonical Regression Guardrails
- canonical overall: baseline 51.00% -> current 49.00% (delta -2.00 pp)
- canonical with_excerpt: baseline 100.00% -> current 96.00% (delta -4.00 pp)
- canonical without_excerpt: baseline 2.00% -> current 2.00% (delta +0.00 pp)
- canonical regressions: 2
- canonical improvements: 0

### Canonical Regression Examples
- drug_id=476 | acetaminophen | status=ambiguous | reason=ambiguous_exact_canonical | matched=None | with_excerpt=True
- drug_id=1196 | amlodipine | status=ambiguous | reason=ambiguous_exact_canonical | matched=None | with_excerpt=True
