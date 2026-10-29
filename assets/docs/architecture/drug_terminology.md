# Drug Terminology Resolution
Last updated: 2026-10-29

## Purpose

DILIGENT resolves every extracted drug mention to a terminology concept before
any LiverTox, DILIrank, or RUCAM evidence is attached. Identity comes from a
locally imported, database-backed terminology. Spelling correction and the
RxNav web service are bounded fallbacks. A clinical run never imports
terminology; it only reads what Knowledge updates stored.

## Sources and Licensing

| Source | Access | License | Role |
| --- | --- | --- | --- |
| RxNorm Current Prescribable Content | Monthly zip (`RXNCONSO`, `RXNREL`), no UMLS login | Public domain (NLM) | Backbone: ingredients (IN), precise ingredients (PIN), multi-ingredient concepts (MIN), brands, synonyms, products (SCD/SBD) and the salt hierarchy. Runs as the `rxnav` update target. |
| DrugCentral | Read-only SQL; default is the public instance `unmtid-dbs.net:5433` (configurable) | CC BY-SA 4.0 | International names, UNII/ChEMBL/INN identifiers, ATC codes, non-US drugs such as metamizole. `struct2parent` is not used because it links prodrugs and metabolites, which are not interchangeable. |
| ChEMBL | REST `molecule.json`, paged, filtered by `max_phase` | CC BY-SA 3.0 | INN/BAN/USAN/JAN, trade names, research codes, salt-to-parent hierarchy. |
| Swissmedic | Monthly "Authorised packaging (human)" XLSX | Open use with source attribution | Swiss brand names and Latin pharmacopoeial ingredient names; product composition. |
| Curated catalog | `data/catalogs/drug_matching.json` (`terminology_aliases`, `error_prone_abbreviations`) | Project-maintained | Abbreviations (HCTZ, MTX, APAP), combination nicknames (co-amoxiclav, TMP/SMX), and abbreviations that must never auto-resolve (MS, MSO4, AZT, HCT, CPZ). |
| RxNav REST | `findRxcuiByString`, `related`, `historystatus` | NLM terms of service | Runtime fallback only; bounded per name and switchable. |
| WHO INN | Not ingested | WHO copyright | WHO offers no open bulk list. INN coverage comes indirectly from ChEMBL `syn_type=INN` and DrugCentral `INN_ID`. |

DrugCentral and ChEMBL are enabled by default. They are downloaded to the local
database by the user's own Knowledge update and are never bundled into release
artifacts. The share-alike terms apply to redistribution of that imported data.
The Knowledge > Terminology view shows each source's license and attribution.

## Schema

Migration `202610290001` extends the existing drug tables instead of adding a
parallel concept store:

- `drugs` is the concept. It gains `concept_type` (`ingredient`,
  `precise_ingredient`, or `multi_ingredient`) and `preferred_inn`. Imports
  never delete concepts, so clinical rows keep their foreign keys.
- `drug_aliases` gains `validation_status` (`verified`, `confirmed`,
  `provisional`, `rejected`), `language`, `country`, `import_run_id` and
  `source_record_id`. Only `verified` and `confirmed` aliases resolve. A
  `rejected` alias hides its (alias, concept) pair from every source.
- `drug_identifiers` gains `source` and `import_run_id`. `(system, value)`
  stays unique; it is the reconciliation anchor for UNII, ChEMBL, DrugCentral,
  InChIKey, and INN ids. RxCUIs stay in `drug_rxnorm_codes`.
- `kb_match_cache` becomes the runtime outcome cache (`resolution_status`,
  `method`, `terminology_revision`, `expires_at`; source `terminology`). Rows
  from the legacy matcher were invalidated by the migration.

New tables:

| Table | Content |
| --- | --- |
| `terminology_import_runs` | One row per source refresh: version, license, mode, status, timings, `stats_json` report, error. The highest completed run id is the terminology revision. |
| `drug_concept_relations` | `form_of` (salt or ester to parent moiety) and `has_component` (combination to ingredient, with position). Stereoisomers are never related. |
| `drug_atc_codes` | ATC codes per concept and source, used to corroborate product compositions. |
| `medicinal_products`, `medicinal_product_ingredients` | RxNorm and Swissmedic products with `composition_status` (`verified` only when every ingredient resolves) and strengths. |
| `terminology_conflicts` | Disagreements found during import: `identifier_mismatch`, `name_identity_mismatch`, `composition_mismatch`. |
| `drug_livertox_links` | Explicit concept-to-monograph links: `monograph_owner`, `verified_alias`, `component_set` (a combination monograph such as Trimethoprim-Sulfamethoxazole linked to the RxNorm combination), `form_of_parent` (a salt inherits its parent's monograph), and `form_of_child` (a parent inherits the monograph of its only linked salt). |

## Import Pipeline

Code: `services/updater/terminology/` (adapters, download, ingestor) and
`repositories/terminology_repository.py` (reconciliation and persistence).

1. Each adapter stages a complete `TerminologySnapshot` without writing to the
   database. Downloads use a `.candidate` file and conditional GET. A
   truncated download, an unreachable source, or a cancellation raises
   `TerminologySourceUnavailable`.
2. The ingestor skips a source whose upstream version equals its last
   completed version, unless a full re-import is forced. Swissmedic has no
   version header, so its version is the release date plus a content hash.
3. Persistence runs in one transaction per source:
   - replace that source's aliases, identifiers, relations, products and ATC
     codes; rows owned by other sources, user decisions (`reviewer`) and
     RxNav runtime aliases (`rxnav_runtime`) are kept;
   - match each staged concept by RxCUI, then identity identifiers, then exact
     name. Contradicting identifiers record a conflict instead of merging,
     and an identifier pointing at a concept unrelated to the exact name is
     overruled by the name;
   - keep a single salt hierarchy rooted at the RxNorm moiety;
   - resolve product ingredients through exact verified aliases with ATC
     corroboration;
   - demote a lower-precedence alias that contradicts a higher-precedence
     source to `provisional` and record a conflict. Precedence is reviewer
     and curated, then RxNorm, RxNav runtime, DrugCentral, ChEMBL,
     Swissmedic. The same brand used for different products in different
     countries stays a legitimate ambiguity;
   - rebuild `drug_livertox_links`.
4. The run row receives the import report and the process-wide fuzzy index is
   invalidated.

A failure or cancellation before commit leaves the previous snapshot in place.
A rerun with unchanged input produces identical rows.

## Runtime Resolution

Code: `services/terminology/` and `repositories/terminology_lookup_repository.py`.

| Stage | Behaviour |
| --- | --- |
| Normalize | Unicode, case and punctuation folding. Strengths (including rates such as `25 mcg/h`), dose forms, routes and release words are kept separately; salt words stay in the key. Combination separators (`/`, `+`, `and`, `und`, `et`) produce component keys. |
| Exact | Indexed lookup on `drug_aliases.alias_norm` for the raw, full and core key. Hits are grouped by moiety. One moiety is `verified`; several are `ambiguous` with candidates. |
| Combination | Each component resolves exactly; the combination concept with the same component set is returned. |
| Error-prone abbreviations | Curated abbreviations such as `MS` never resolve; the concepts they could denote are returned as candidates. |
| Language variants | Romance and Latin endings (metformina, omeprazolo, paracetamolum) and an orthographic skeleton (ph/f, y/i, doubled letters) map to exact aliases. |
| Unverified product | A brand whose composition could not be verified returns `probable` with `accepted=false`. |
| Fuzzy | A trigram shortlist (default 50) scored with Damerau-Levenshtein (OSA). Accepted only when exactly one concept is within the length-scaled edit bound (1 below 8 characters, otherwise 2), no other concept is within that bound, the runner-up is at least the margin farther away, and the first letter matches. An accepted correction is `probable` and flagged for review. Abbreviations and research codes are never fuzzy-matched. |
| RxNav fallback | Only after the local stages fail and when enabled. A budget of requests per name (default 4). A unique ingredient or combination mapped by RxCUI to a local concept is persisted as a verified `rxnav_runtime` alias. A clean miss is cached for `external_negative_cache_days`; timeouts and budget stops are not cached. |

Each result is a `TerminologyResolution`: original and normalized input,
concept and ingredients (with parent salts), status (`verified`, `probable`,
`ambiguous`, `unresolved`), `accepted`, method, matched alias, source,
candidates (top 5 with edit distance and a raw similarity that is a ranking
signal, not a probability), review reasons, and whether the stated strength
exists in a known product. Results are cached per process by raw input and
terminology revision.

## Clinical Integration

`DrugResolutionService` resolves each extracted mention and attaches LiverTox
evidence only through `drug_livertox_links`:

| Decision status | Meaning |
| --- | --- |
| `accepted_verified` | Verified identity with a linked monograph. |
| `accepted_probable` | Spelling-corrected identity with a linked monograph; requires review. |
| `missing_livertox` | Accepted identity without a linked monograph (for example hydrochlorothiazide, which LiverTox covers only in a class monograph). |
| `ambiguous_requires_review` | Several candidate concepts; no evidence is attached. |
| `unresolved` | No accepted identity, including products with unverified composition. |
| `rejected_false_positive` | Laboratory or non-drug text. |

A combination without its own monograph is split into its ingredients so each
keeps its own evidence. The configured LLM may propose identities for
unresolved mentions; a proposal is used only if it resolves to a `verified`
local identity with a linked monograph. Match-audit issues:
`drug_identity_unresolved`, `drug_identity_ambiguous`,
`drug_identity_probable`, `drug_composition_unverified`,
`livertox_link_missing`. The final report starts with a "Drug identification"
table (entered name, resolved ingredients, identity status, source).

## Reviewer Decisions

Knowledge > Terminology and Clinical Sessions > Detected drugs let a reviewer
confirm a candidate or reject a mapping. Decisions are stored as aliases with
source `reviewer` and status `confirmed` or `rejected`, so no source refresh
replaces them. They apply to future assessments; stored sessions are not
rewritten. A rejection hides an exact alias mapping; it does not change fuzzy
candidates.

## Population and Refresh

- Knowledge > Update all runs RxNav (RxNorm and curated aliases), LiverTox,
  Terminology (DrugCentral, ChEMBL, Swissmedic, curated), and DILIrank in that
  order. Each target can also run alone; the Terminology dialog can skip a
  source for one run or force a full re-import.
- The LiverTox update rebuilds monograph links at its end.
- A first full population downloads the RxNorm Prescribable zip (about
  70 MB) and the Swissmedic workbook (about 3 MB), pages about 4,200 approved
  ChEMBL molecules over REST, and queries the public DrugCentral database.
  On the development machine each source persisted in 15 to 80 seconds.

## Configuration

| Setting (Settings page) | Default | Effect |
| --- | --- | --- |
| Drug Matching: minimum spelling-correction length | 5 | Shorter inputs are only matched exactly. |
| Drug Matching: short-name threshold / edit bounds | 8 / 1 / 2 | Length-scaled edit bound. |
| Drug Matching: runner-up margin | 1 | Extra edits required of the next-closest concept. |
| Drug Matching: candidate shortlist size | 50 | Aliases scored per input. |
| Drug Matching: resolution cache size | 5000 | Per-process cached resolutions. |
| Drug Matching: RxNav fallback, requests per name, negative result lifetime | on, 4, 7 days | Runtime fallback. |
| Integrations: terminology download timeout | 300 s | RxNorm, ChEMBL and Swissmedic downloads. |
| Integrations: DrugCentral (enabled, host, port, database, user) | on, public instance | DrugCentral source. |
| Integrations: ChEMBL (enabled, clinical phase) | on, 4 | ChEMBL source; phase 4 means approved. |
| Integrations: Swissmedic | on | Swissmedic source. |

The fuzzy defaults were kept after calibration on the benchmark: no grid point
changed the number of wrong automatic matches.

## Inspection API

`/api/inspection/terminology/...`: `summary`, `runs`, `conflicts`, `resolve`,
`aliases/confirm`, `aliases/reject`, `update-config`, and `jobs` (start,
status, cancel). See [api_surface.md](api_surface.md).

## Benchmark

`app/tests/benchmarks/drug_terminology/` contains 407 labeled cases (generics,
US and Swiss brands, INN/USAN pairs, language variants, typos, abbreviations,
salts, stereoisomers, look-alike pairs, strengths, unknowns) and the harness.
The labels are author-curated from standard nomenclature and are not a
clinical validation. Latest results:
[`assets/QA/drug-terminology-20261029/report.md`](../../QA/drug-terminology-20261029/report.md).

## Known Limitations

- WHO INN lists cannot be ingested.
- The public DrugCentral instance can be unavailable, and its RxNorm cross-walk
  contains errors; the exact-name check overrules contradicting identifiers.
- RxNorm Prescribable Content omits obsolete and non-US names; DrugCentral,
  ChEMBL and Swissmedic fill part of the gap. Some brands are in no source
  (for example Zurcal, Tazocin, Dafalgan Codein) and stay unresolved.
- About 23% of Swissmedic products keep an unverified composition because a
  Latin ingredient name does not resolve; their brands get no alias.
- Class terms such as "insulin" or "vitamin D" resolve to LiverTox class
  concepts.
- Multilingual coverage is limited to source names, Swiss products, Latin
  rules and curated Romance endings.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Every drug is unresolved | Knowledge > Terminology shows no completed RxNorm run; run Update all. |
| DrugCentral import failed | The public instance may be down or blocked by a firewall (port 5433). Disable DrugCentral for the run or point Integrations to a local restore. |
| A brand resolves to the wrong drug | Reject the mapping in Knowledge > Terminology and confirm the correct concept; check Open conflicts for the source that introduced it. |
| A name is "Probable — review" | It was spelling-corrected. Confirm it once to make it verified. |
| A drug has no LiverTox evidence | The identity may be `missing_livertox`: LiverTox has no monograph linked to that concept, or only a class monograph. |
