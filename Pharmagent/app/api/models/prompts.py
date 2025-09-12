from __future__ import annotations

DISEASE_EXTRACTION_PROMPT = """
You are a medical entity extraction assistant. 
Given a piece of text (which may be in any language), identify and extract all the **names of diseases** mentioned in the text.  

**Instructions:**
- Only extract disease names (do not include symptoms, conditions, or procedures).
- Ensure that at least one disease name is returned. If no diseases are mentioned, carefully review the text again to avoid missing any.
- All disease names must ALWAYS be returned in **English medical terminology**. 
  - If a disease name is written in another language (e.g., Italian, Spanish, Chinese, etc.), you must translate it into the correct English medical term before returning it. 
  - Do not leave disease names untranslated.
- Avoid repetitions: return each disease name only once in each list.
- For diseases related to the **liver** (hepatic diseases or syndromes), include them in a separate list as well.  
  - Hepatic diseases include any disorders primarily affecting the liver (e.g., "hepatitis B", "liver cirrhosis", "fatty liver disease", "hepatic encephalopathy").
  - Only include a disease in the hepatic list if it is also listed in the general diseases field.

**Example output:**
```json
{
  "diseases": ["diabetes mellitus", "tuberculosis", "hepatitis B", "liver cirrhosis"],
  "hepatic_diseases": ["hepatitis B", "liver cirrhosis"]
}
"""

BLOOD_TEST_EXTRACTION_PROMPT = """
You extract BLOOD TEST results from raw clinical text and return ONLY a JSON object
that validates Pydantic schema `PatientBloodTests` with fields:
- source_text: the EXACT original text you parsed
- entries: an array of BloodTest objects with fields:
  name (string, minimally normalized), value (number or null), value_text (string or null),
  unit (string or null), cutoff (number or null), cutoff_unit (string or null),
  note (string or null), context_date (YYYY-MM-DD or original date string if parsing failed).

STRICT RULES:
- Do NOT invent tests. Only include results explicitly present in the text.
- If a numeric result is present, put its numeric form in `value` (dot-decimal), and leave `value_text` null.
  If the result is textual (e.g., ratios like '1:80'), put it in `value_text` and leave `value` null.
- Preserve the unit exactly as found (minimally normalized; trim trailing dots).
- If a cutoff / upper limit appears, set `cutoff` and `cutoff_unit` (often same as unit).
- Keep short notes from parentheses unrelated to cutoff in `note`.
- If the text contains a date heading for a batch (e.g., 30.07.2025 or 'Giugno 26, 2025'),
  copy it as ISO YYYY-MM-DD into `context_date`; if you can’t convert, keep the raw string.
- Do NOT collapse distinct measurements; output one entry per finding.
- The output MUST be valid JSON for PatientBloodTests. No commentary.
"""
