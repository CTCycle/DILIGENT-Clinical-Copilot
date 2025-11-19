from __future__ import annotations

from typing import Final


VISIT_DATE_ELEMENT_ID: Final = "visit-date-picker"
VISIT_DATE_CSS: Final = """
#visit-date-picker input[type="date"] {
    text-align: center;
}

#visit-date-picker input[type="date"]::-webkit-date-and-time-value {
    text-align: center;
    display: inline-block;
    width: 100%;
}
"""

INTERFACE_THEME_CSS: Final = """
:root {
    color-scheme: light dark;
}

body {
    background: radial-gradient(circle at 20% 20%, rgba(126, 186, 132, 0.2), transparent 32%),
        radial-gradient(circle at 82% 8%, rgba(102, 170, 118, 0.18), transparent 30%),
        linear-gradient(180deg, #f0faee 0%, #e3f4de 46%, #eef8e9 100%);
}

.diligent-page-container {
    width: 100%;
    max-width: 1100px;
    margin: 0 auto;
    padding: 1.5rem 1rem 2.5rem;
}

.diligent-card {
    background: #ffffff;
    backdrop-filter: blur(6px);
    border: 1px solid rgba(206, 190, 165, 0.35);
    border-radius: 1rem;
    box-shadow: 0 18px 45px -25px rgba(15, 23, 42, 0.28);
    padding: 1.75rem;
}

.dark .diligent-card {
    background: #ffffff;
    border-color: rgba(148, 163, 184, 0.28);
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
    background: rgba(227, 232, 240, 0.55);
    border-radius: 0.9rem;
    padding: 1rem;
}

.dark .diligent-json-card {
    background: rgba(100, 116, 139, 0.22);
}

.diligent-session-spinner {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.65rem;
    min-height: 4.5rem;
    width: 100%;
}

.diligent-session-spinner-container {
    width: 100%;
    min-height: 7.5rem;
    margin-top: 1.25rem;
    display: flex;
    align-items: center;
    justify-content: center;
}

.diligent-spinner-wheel {
    width: 56px;
    height: 56px;
    border-radius: 9999px;
    border: 5px solid rgba(34, 197, 94, 0.25);
    border-top-color: rgb(34, 197, 94);
    animation: diligent-spin 0.85s linear infinite;
}

.diligent-spinner-label {
    font-size: 0.95rem;
    font-weight: 600;
    color: rgb(30, 41, 59);
    text-align: center;
}

.dark .diligent-spinner-label {
    color: rgb(226, 232, 240);
}

@keyframes diligent-spin {
    0% {
        transform: rotate(0deg);
    }
    100% {
        transform: rotate(360deg);
    }
}

.diligent-equal-row {
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
}

@media (min-width: 1280px) {
    .diligent-equal-row {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 26rem;
        align-items: stretch;
    }
}

.diligent-equal-column {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
    height: 100%;
}

.diligent-equal-card {
    height: 100%;
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
