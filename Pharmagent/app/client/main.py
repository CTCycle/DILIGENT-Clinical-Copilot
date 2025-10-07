from __future__ import annotations

from typing import Final

import gradio as gr

from Pharmagent.app.client.controllers import (
    clear_agent_fields,
    normalize_visit_date_component,
    preload_selected_models,
    pull_selected_models,
    run_agent,
    set_agent_model,
    set_cloud_model,
    set_llm_provider,
    set_ollama_reasoning,
    set_ollama_temperature,
    set_parsing_model,
    start_ollama_client,
    toggle_cloud_services,
)
from Pharmagent.app.configurations import ClientRuntimeConfig
from Pharmagent.app.constants import (
    AGENT_MODEL_CHOICES,
    CLOUD_MODEL_CHOICES,
    CLOUD_PROVIDERS,
    PARSING_MODEL_CHOICES,
)


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


###############################################################################
def _noop() -> None:
    return None


###############################################################################
def create_interface() -> gr.Blocks:
    provider = ClientRuntimeConfig.get_llm_provider()
    cloud_models = CLOUD_MODEL_CHOICES.get(provider, [])
    selected_cloud_model = ClientRuntimeConfig.get_cloud_model()
    if selected_cloud_model not in cloud_models:
        selected_cloud_model = cloud_models[0] if cloud_models else ""
        ClientRuntimeConfig.set_cloud_model(selected_cloud_model)
    with gr.Blocks(
        title="Pharmagent Clinical Copilot",
        analytics_enabled=False,
        theme="soft",
        css=VISIT_DATE_CSS,
    ) as demo:
        gr.Markdown("## Pharmagent Clinical Copilot")

        with gr.Row():
            with gr.Column(scale=3):
                anamnesis = gr.Textbox(
                    label="Anamnesis",
                    placeholder="Enter anamnesis details...",
                    lines=10,
                    max_lines=100,
                )
                has_diseases = gr.Checkbox(
                    label="Has hepatic diseases",
                    value=False,
                )
                symptoms = gr.CheckboxGroup(
                    label="Observed symptoms",
                    choices=["Hitterus", "Pain", "Scretching"],
                )
                drugs = gr.Textbox(
                    label="Current Drugs",
                    placeholder="List current therapies, dosage and schedule...",
                    lines=10,
                    max_lines=100,
                )
                exams = gr.Textbox(
                    label="Additional Exams",
                    placeholder="Provide lab or imaging exam notes...",
                    lines=10,
                    max_lines=100,
                )
                with gr.Row():
                    alt = gr.Textbox(
                        label="ALT",
                        placeholder="e.g., 189 or 189 U/L",
                        lines=1,
                        scale=3,
                    )
                    alt_max = gr.Textbox(
                        label="ALT Max",
                        placeholder="e.g., 47 U/L",
                        lines=1,
                        scale=2,
                    )
                with gr.Row():
                    alp = gr.Textbox(
                        label="ALP",
                        placeholder="e.g., 140 or 140 U/L",
                        lines=1,
                        scale=3,
                    )
                    alp_max = gr.Textbox(
                        label="ALP Max",
                        placeholder="e.g., 150 U/L",
                        lines=1,
                        scale=2,
                    )

            with gr.Column(scale=1):
                patient_name = gr.Textbox(
                    label="Patient Name",
                    placeholder="e.g., Marco Rossi",
                    lines=1,
                )
                visit_date = gr.DateTime(
                    label="Visit Date",
                    include_time=False,
                    type="datetime",
                    value=None,
                    elem_id=VISIT_DATE_ELEMENT_ID,
                )
                visit_date.change(
                    fn=normalize_visit_date_component,
                    inputs=visit_date,
                    outputs=visit_date,
                )
                process_from_files = gr.Checkbox(
                    label="Process patients from files",
                    value=False,
                )
                translate_to_eng = gr.Checkbox(
                    label="Always translate to English",
                    value=False,
                )
                with gr.Column():
                    run_button = gr.Button("Run Workflow", variant="primary")
                    clear_button = gr.Button("Clear all")
                with gr.Accordion("Runtime Configuration", open=False):
                    with gr.Column():
                        with gr.Row():
                            with gr.Column(scale=1):
                                use_cloud_services = gr.Checkbox(
                                    label="Use Cloud Services",
                                    value=ClientRuntimeConfig.is_cloud_enabled(),
                                )
                                llm_provider_dropdown = gr.Dropdown(
                                    label="Cloud Service",
                                    choices=CLOUD_PROVIDERS,
                                    value=provider,
                                    interactive=ClientRuntimeConfig.is_cloud_enabled(),
                                )
                            cloud_model_dropdown = gr.Dropdown(
                                label="Cloud Model",
                                choices=cloud_models,
                                value=selected_cloud_model,
                                interactive=ClientRuntimeConfig.is_cloud_enabled(),
                            )
                        with gr.Row():
                            with gr.Column(scale=1):
                                parsing_model_dropdown = gr.Dropdown(
                                    label="Parsing Model",
                                    choices=PARSING_MODEL_CHOICES,
                                    value=ClientRuntimeConfig.get_parsing_model(),
                                )
                                agent_model_dropdown = gr.Dropdown(
                                    label="Agent Model",
                                    choices=AGENT_MODEL_CHOICES,
                                    value=ClientRuntimeConfig.get_agent_model(),
                                )
                                temperature_input = gr.Number(
                                    label="Temperature",
                                    value=ClientRuntimeConfig.get_ollama_temperature(),
                                    minimum=0.0,
                                    maximum=2.0,
                                    step=0.1,
                                    interactive=not ClientRuntimeConfig.is_cloud_enabled(),
                                )
                                reasoning_checkbox = gr.Checkbox(
                                    label="Enable reasoning (think)",
                                    value=ClientRuntimeConfig.is_ollama_reasoning_enabled(),
                                    interactive=not ClientRuntimeConfig.is_cloud_enabled(),
                                )
                                pull_models_button = gr.Button(
                                    "Pull models",
                                    variant="secondary",
                                )
                                start_ollama_button = gr.Button(
                                    "Start Ollama client",
                                    variant="secondary",
                                    interactive=not ClientRuntimeConfig.is_cloud_enabled(),
                                )
                                preload_button = gr.Button(
                                    "Preload models",
                                    variant="secondary",
                                    interactive=not ClientRuntimeConfig.is_cloud_enabled(),
                                )

        output = gr.Textbox(
            label="Agent Output",
            lines=30,
            show_copy_button=True,
            interactive=False,
        )

        use_cloud_services.change(
            fn=toggle_cloud_services,
            inputs=use_cloud_services,
            outputs=[
                llm_provider_dropdown,
                cloud_model_dropdown,
                start_ollama_button,
                preload_button,
                temperature_input,
                reasoning_checkbox,
            ],
        )
        llm_provider_dropdown.change(
            fn=set_llm_provider,
            inputs=llm_provider_dropdown,
            outputs=[llm_provider_dropdown, cloud_model_dropdown],
        )
        cloud_model_dropdown.change(
            fn=set_cloud_model,
            inputs=cloud_model_dropdown,
            outputs=cloud_model_dropdown,
        )
        parsing_model_dropdown.change(
            fn=set_parsing_model,
            inputs=parsing_model_dropdown,
            outputs=parsing_model_dropdown,
        )
        agent_model_dropdown.change(
            fn=set_agent_model,
            inputs=agent_model_dropdown,
            outputs=agent_model_dropdown,
        )
        temperature_input.change(
            fn=set_ollama_temperature,
            inputs=temperature_input,
            outputs=temperature_input,
        )
        reasoning_checkbox.change(
            fn=set_ollama_reasoning,
            inputs=reasoning_checkbox,
            outputs=reasoning_checkbox,
        )

        pull_models_button.click(
            fn=pull_selected_models,
            inputs=[parsing_model_dropdown, agent_model_dropdown],
            outputs=output,
        )

        run_button.click(
            fn=run_agent,
            inputs=[
                patient_name,
                visit_date,
                anamnesis,
                has_diseases,
                drugs,
                exams,
                alt,
                alt_max,
                alp,
                alp_max,
                symptoms,
                process_from_files,
                translate_to_eng,
            ],
            outputs=output,
            api_name="run_agent",
        )
        start_ollama_button.click(
            fn=start_ollama_client,
            outputs=output,
        )
        preload_button.click(
            fn=preload_selected_models,
            inputs=[parsing_model_dropdown, agent_model_dropdown],
            outputs=output,
        )
        clear_button.click(
            fn=clear_agent_fields,
            outputs=[
                patient_name,
                visit_date,
                anamnesis,
                drugs,
                exams,
                alt,
                alt_max,
                alp,
                alp_max,
                symptoms,
                process_from_files,
                translate_to_eng,
                has_diseases,
                output,
            ],
        )

        demo.load(
            fn=_noop,
            inputs=None,
            outputs=None,
            js=VISIT_DATE_LOCALE_JS,
        )

    return demo


###############################################################################
def launch_interface() -> None:
    create_interface().queue(max_size=32).launch(
        server_name="127.0.0.1",
        server_port=7861,
        inbrowser=True,
    )


if __name__ == "__main__":
    launch_interface()
