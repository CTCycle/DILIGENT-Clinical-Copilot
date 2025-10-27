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

INTERFACE_THEME_CSS: Final = """
:root {
    color-scheme: light dark;
}

body {
    background: radial-gradient(circle at top, rgba(23, 162, 184, 0.08), transparent 55%),
        radial-gradient(circle at bottom, rgba(99, 102, 241, 0.08), transparent 45%),
        var(--nicegui-background);
}

.diligent-page-container {
    width: 100%;
    max-width: 1100px;
    margin: 0 auto;
    padding: 1.5rem 1rem 2.5rem;
}

.diligent-card {
    background: rgba(255, 255, 255, 0.92);
    backdrop-filter: blur(6px);
    border-radius: 1rem;
    box-shadow: 0 18px 45px -25px rgba(15, 23, 42, 0.35);
    padding: 1.75rem;
}

.dark .diligent-card {
    background: rgba(15, 23, 42, 0.78);
    box-shadow: 0 18px 45px -30px rgba(15, 23, 42, 0.85);
}

.diligent-card-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: rgb(30, 41, 59);
    margin-bottom: 1rem;
}

.dark .diligent-card-title {
    color: rgb(226, 232, 240);
}

.diligent-subtitle {
    font-size: 1rem;
    font-weight: 600;
    color: rgb(71, 85, 105);
    margin-bottom: 0.5rem;
}

.dark .diligent-subtitle {
    color: rgb(203, 213, 225);
}

.diligent-json-card {
    background: rgba(148, 163, 184, 0.12);
    border-radius: 0.9rem;
    padding: 1rem;
}

.dark .diligent-json-card {
    background: rgba(100, 116, 139, 0.22);
}
"""

PAGE_CONTAINER_CLASSES: Final = "diligent-page-container flex flex-col gap-6"
CARD_BASE_CLASSES: Final = "diligent-card w-full flex flex-col gap-4"
JSON_CARD_CLASSES: Final = "diligent-json-card w-full flex flex-col gap-3"

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
