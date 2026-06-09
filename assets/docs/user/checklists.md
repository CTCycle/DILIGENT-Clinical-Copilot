# User Checklists
Last updated: 2026-06-03

## Recommended End-to-end Journey
1. Start DILIGENT with `start_on_windows.bat`.
2. Confirm the UI opens.
3. Confirm backend health at `/api/health`.
4. Open **Model Configurations**.
5. Select provider and model.
6. Add and activate any required access key.
7. Open **DILI Agent**.
8. Enter patient context, medication exposure, labs, symptoms, and timing.
9. Run the assessment.
10. Review the generated report.
11. Correct or enrich the input if needed.
12. Re-run if needed.
13. Copy the final reviewed output.
14. Open **Clinical Sessions** to confirm the session was saved.
15. Use **Patient Timeline** or **Data Inspection** for supporting review when needed.

## Good Input Checklist
- patient age or relevant demographic context
- suspected drug or exposure
- exposure start date or approximate timing
- symptom onset date or approximate timing
- ALT, AST, ALP, bilirubin, and INR if available
- relevant baseline labs if available
- relevant comorbidities
- alternative causes if known
- dechallenge or rechallenge information if available
- current clinical question

## Output Review Checklist
- verify all dates
- verify lab values
- verify units and reference ranges
- verify medication names
- verify that alternative causes were considered
- verify that the model did not invent facts
- verify that the conclusion follows from the entered evidence
- add human clinical interpretation before formal use

## What Not To Do
- Do not treat generated output as a final diagnosis.
- Do not enter real patient data into cloud workflows without approval.
- Do not share provider access keys.
- Do not manually edit the local database while the app is running.
- Do not ignore backend or provider errors.
- Do not assume saved sessions exist unless the assessment completed successfully.
- Do not copy model output into clinical documentation without human review.

## Where To Go Next
- Installation and developer setup: `README.md`
- Architecture, runtime, coding, and UI reference: `assets/docs/project_index.md`
