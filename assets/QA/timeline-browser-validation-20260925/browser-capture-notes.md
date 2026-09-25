# Browser capture notes — 2026-09-25

Rendered captures were inspected in the Codex in-app Browser at `1280 × 720`.
The desktop-width warning was absent.

| Capture | Visible state |
|---|---|
| 2B provenance | Patient Timeline header showed `Fallback chronology`, `Invalid structured provider response`, `qwen3.5:2b`, three clinical events, and January 2025 coverage. |
| 2B event inspector | Therapy, symptom, and ALT cards were visible; the symptom inspector showed `Day precision · explicit`, normalized date `2025-01-17`, `fallback_parser`, and the exact source quote. |
| 9B chronology | Patient Timeline showed `LLM generated`, `qwen3.5:9b`, three events, source labels `drugs`, `laboratory_analysis`, and `anamnesis`, and month/day placements. |
| 9B event inspector | Acetaminophen inspector showed month precision, normalized date `2025-01`, direct source evidence, and the uncertainty note that the source did not specify a day. |
| Reload checks | Both timeline URLs reloaded to the same persisted status, exact model, event count, evidence labels, and date precision. |

The browser surface emitted the captures inline during validation but did not
provide a durable binary screenshot export API. The observations above are the
sanitized durable record; the rendered states were directly inspected rather
than inferred from API output.
