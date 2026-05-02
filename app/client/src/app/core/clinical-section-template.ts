export const DILI_CLINICAL_INPUT_TEMPLATE = `Use a structured format so the text can be split safely.

Markdown:
# Anamnesis
Patient history block one.

# Current therapy
Current therapy block.

# Anamnesis
Patient history block two.

# Laboratory analysis
Laboratory analysis block.

XML:
<anamnesis>Patient history block one.</anamnesis>
<current_therapy>Current therapy block.</current_therapy>
<anamnesis>Patient history block two.</anamnesis>
<laboratory_analysis>Laboratory analysis block.</laboratory_analysis>

JSON:
{
  "anamnesis": ["Patient history block one.", "Patient history block two."],
  "current_therapy": ["Current therapy block."],
  "laboratory_analysis": ["Laboratory analysis block."]
}`;
