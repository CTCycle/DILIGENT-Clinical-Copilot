from __future__ import annotations

from typing import Final


VISIT_DATE_ELEMENT_ID: Final = "visit-date-picker"
VISIT_DATE_CSS: Final = """
#visit-date-picker input[type="date"]::-webkit-datetime-edit {
    display: flex;
}

#visit-date-picker input[type="date"]::-webkit-datetime-edit-fields-wrapper {
    display: flex;
}

#visit-date-picker input[type="date"]::-webkit-datetime-edit-day-field {
    order: 1;
}

#visit-date-picker input[type="date"]::-webkit-datetime-edit-text {
    order: 2;
}

#visit-date-picker input[type="date"]::-webkit-datetime-edit-month-field {
    order: 3;
}

#visit-date-picker input[type="date"]::-webkit-datetime-edit-year-field {
    order: 5;
}

#visit-date-picker input[type="date"]::-webkit-datetime-edit-text:last-of-type {
    order: 4;
}
"""

VISIT_DATE_LOCALE_JS: Final = f"""
() => {{
    const container = document.querySelector('#{VISIT_DATE_ELEMENT_ID}');
    if (!container) {{
        return;
    }}
    const input = container.querySelector('input[type="date"]');
    if (!input) {{
        return;
    }}
    input.setAttribute('lang', 'en-GB');
}}
"""
