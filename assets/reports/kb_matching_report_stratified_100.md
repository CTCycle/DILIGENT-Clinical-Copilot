# KB Matching Evaluation (Stratified 100)

## Design
- Drugs tested: 100
- With excerpt drugs: 50
- Without excerpt drugs: 50
- Inputs per drug: 3
- Total inputs: 300
- Alias filtering: preferred alias kinds + generic token/short alias exclusion + collision-aware selection

## Overall
- Successes: 125
- Match rate: 41.67%

## By Input Type
- canonical: 51/100 (51.0%)
- real_alias: 49/100 (49.0%)
- malformed_alias: 25/100 (25.0%)

## By Stratum
- with_excerpt: 123/150 (82.0%)
- without_excerpt: 2/150 (1.33%)

## Input Type x Stratum
- canonical | with_excerpt: 50/50 (100.0%) | without_excerpt: 1/50 (2.0%)
- real_alias | with_excerpt: 48/50 (96.0%) | without_excerpt: 1/50 (2.0%)
- malformed_alias | with_excerpt: 25/50 (50.0%) | without_excerpt: 0/50 (0.0%)

## Top Failure Patterns
- malformed_alias | missing | no_match: 71
- canonical | missing | no_match: 37
- real_alias | ambiguous | ambiguous_exact_alias: 35
- real_alias | missing | no_match: 13
- canonical | ambiguous | ambiguous_exact_alias: 9
- malformed_alias | ambiguous | ambiguous_fuzzy: 3
- real_alias | matched | exact_canonical: 2
- canonical | matched | fuzzy: 2
- real_alias | matched | fuzzy: 1
- malformed_alias | ambiguous | ambiguous_exact_alias: 1
- canonical | ambiguous | ambiguous_fuzzy: 1

## Real Alias Failure Patterns
- ambiguous | ambiguous_exact_alias: 35
- missing | no_match: 13
- matched | exact_canonical: 2
- matched | fuzzy: 1

## Real Alias Failure Examples
- acoramidis | alias=Attruby | status=matched | reason=exact_canonical | matched=Attruby
- alprazolam | alias=Disintegrating | status=ambiguous | reason=ambiguous_exact_alias | matched=None
- 10 ML sodium chloride 9 MG/ML Injection | alias=Normal Saline | status=missing | reason=no_match | matched=None
- 10 ML sodium chloride 9 MG/ML Prefilled Syringe | alias=NaCl | status=ambiguous | reason=ambiguous_exact_alias | matched=None
- 100 ML sodium chloride 4.5 MG/ML Injection | alias=NaCl | status=ambiguous | reason=ambiguous_exact_alias | matched=None
- 100 ML sodium chloride 9 MG/ML Injection | alias=NaCl | status=ambiguous | reason=ambiguous_exact_alias | matched=None
- 1000 ML potassium chloride 0.02 MEQ/ML / sodium chloride 4.5 MG/ML Injection | alias=Pot NaCl | status=missing | reason=no_match | matched=None
- 1000 ML potassium chloride 0.02 MEQ/ML / sodium chloride 9 MG/ML Injection | alias=Pot NaCl | status=missing | reason=no_match | matched=None
- 1000 ML potassium chloride 0.04 MEQ/ML / sodium chloride 9 MG/ML Injection | alias=Pot NaCl | status=missing | reason=no_match | matched=None
- 1000 ML sodium chloride 4.5 MG/ML Injection | alias=NaCl | status=ambiguous | reason=ambiguous_exact_alias | matched=None
- 1000 ML sodium chloride 9 MG/ML Injection | alias=NaCl | status=ambiguous | reason=ambiguous_exact_alias | matched=None
- 150 ML sodium chloride 9 MG/ML Injection | alias=NaCl | status=ambiguous | reason=ambiguous_exact_alias | matched=None
- 2 ML sodium chloride 9 MG/ML Injection | alias=NaCl | status=ambiguous | reason=ambiguous_exact_alias | matched=None
- 20 ML sodium chloride 146 MG/ML Injection | alias=NaCl | status=ambiguous | reason=ambiguous_exact_alias | matched=None
- 20 ML sodium chloride 9 MG/ML Injection | alias=NaCl | status=ambiguous | reason=ambiguous_exact_alias | matched=None
- 25 ML sodium chloride 4.5 MG/ML Injection | alias=NaCl | status=ambiguous | reason=ambiguous_exact_alias | matched=None
- 25 ML sodium chloride 9 MG/ML Injection | alias=NaCl | status=ambiguous | reason=ambiguous_exact_alias | matched=None
- 250 ML sodium chloride 4.5 MG/ML Injection | alias=NaCl | status=ambiguous | reason=ambiguous_exact_alias | matched=None
- 250 ML sodium chloride 9 MG/ML Injection | alias=NaCl | status=ambiguous | reason=ambiguous_exact_alias | matched=None
- 30 ML sodium chloride 234 MG/ML Injection | alias=NaCl | status=ambiguous | reason=ambiguous_exact_alias | matched=None

## Optimization Recommendations
- alias uniqueness weighting and collision penalty
- ambiguity margin threshold before accepting top-1
- generic alias blocklist expansion
- dose/form stripping fallback for formulation aliases
- evaluate dual KPI: excerpt-covered and mixed catalog

## Prompt Embedding
Use this Markdown report plus the JSON file for optimization prompts.