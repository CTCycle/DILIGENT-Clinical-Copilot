from __future__ import annotations

from typing import Final

import gradio as gr

from DILIGENT.app.client.controllers import (
    clear_session_fields,
    normalize_visit_date_component,
    pull_selected_models,
    run_DILI_session,
    set_clinical_model,
    set_cloud_model,
    set_llm_provider,
    set_ollama_reasoning,
    set_ollama_temperature,
    set_parsing_model,
    toggle_cloud_services,
)
from DILIGENT.app.configurations import ClientRuntimeConfig
from DILIGENT.app.constants import (
    CLINICAL_MODEL_CHOICES,
    CLOUD_MODEL_CHOICES,
    CLOUD_PROVIDERS,
    PARSING_MODEL_CHOICES,
)

from DILIGENT.app.client.layouts import (
    VISIT_DATE_ELEMENT_ID,
    VISIT_DATE_CSS,
    VISIT_DATE_LOCALE_JS,
)


###############################################################################
def create_interface() -> gr.Blocks:
    provider = ClientRuntimeConfig.get_llm_provider()
    cloud_models = CLOUD_MODEL_CHOICES.get(provider, [])
    selected_cloud_model = ClientRuntimeConfig.get_cloud_model()
    if selected_cloud_model not in cloud_models:
        selected_cloud_model = cloud_models[0] if cloud_models else ""
        ClientRuntimeConfig.set_cloud_model(selected_cloud_model)
    with gr.Blocks(
        title="DILIGENT Clinical Copilot",
        analytics_enabled=False,
        theme="soft",
        css=VISIT_DATE_CSS,
    ) as demo:
        gr.Markdown("## DILIGENT Clinical Copilot")

        with gr.Row():
            with gr.Column(scale=3):
                anamnesis = gr.Textbox(
                    label="Anamnesis and Exams",
                    placeholder="Describe the clinical picture, including exams if relevant...",
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
                with gr.Column():
                    run_button = gr.Button("Run Workflow", variant="primary")
                    export_button = gr.DownloadButton(
                        "Download report",
                        value=None,
                        interactive=False,
                    )
                    clear_button = gr.Button("Clear all")
                with gr.Accordion("Session Configurations", open=False):
                    use_rag_checkbox = gr.Checkbox(
                        label="Use Retrieval Augmented Generation (RAG)",
                        value=False,
                    )
                with gr.Accordion("Model Configurations", open=False):
                    with gr.Column():
                        use_cloud_services = gr.Checkbox(
                            label="Use Cloud Services",
                            value=ClientRuntimeConfig.is_cloud_enabled(),
                        )
                        with gr.Row():
                            with gr.Column(scale=1):
                                with gr.Group():
                                    gr.Markdown("**Cloud Configuration**")
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
                            with gr.Column(scale=1):
                                with gr.Group():
                                    gr.Markdown("**Ollama Configuration**")
                                    parsing_model_dropdown = gr.Dropdown(
                                        label="Parsing Model",
                                        choices=PARSING_MODEL_CHOICES,
                                        value=ClientRuntimeConfig.get_parsing_model(),
                                    )
                                    clinical_model_dropdown = gr.Dropdown(
                                        label="Clinical Model",
                                        choices=CLINICAL_MODEL_CHOICES,
                                        value=ClientRuntimeConfig.get_clinical_model(),
                                        interactive=not ClientRuntimeConfig.is_cloud_enabled(),
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
                                        interactive=not ClientRuntimeConfig.is_cloud_enabled(),
                                    )
                json_output = gr.JSON(
                    label="Agent Output (JSON)",
                    value=None,
                    visible=False,
                )

        gr.Markdown("### Agent Output")
        markdown_output = gr.Markdown(
            value="",
            render=False,
        )
        markdown_output.render()

        use_cloud_services.change(
            fn=toggle_cloud_services,
            inputs=use_cloud_services,
            outputs=[
                llm_provider_dropdown,
                cloud_model_dropdown,
                pull_models_button,
                temperature_input,
                reasoning_checkbox,
                clinical_model_dropdown,
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
        clinical_model_dropdown.change(
            fn=set_clinical_model,
            inputs=clinical_model_dropdown,
            outputs=clinical_model_dropdown,
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
            inputs=[
                parsing_model_dropdown,
                clinical_model_dropdown,
            ],
            outputs=[markdown_output, json_output],
        )

        run_button.click(
            fn=run_DILI_session,
            inputs=[
                patient_name,
                visit_date,
                anamnesis,
                has_diseases,
                drugs,
                alt,
                alt_max,
                alp,
                alp_max,
                symptoms,
                use_rag_checkbox,
            ],
            outputs=[markdown_output, json_output, export_button],
            api_name="run_DILI_session",
        )
        clear_button.click(
            fn=clear_session_fields,
            outputs=[
                patient_name,
                visit_date,
                anamnesis,
                drugs,
                alt,
                alt_max,
                alp,
                alp_max,
                symptoms,
                has_diseases,
                use_rag_checkbox,
                markdown_output,
                json_output,
                export_button,
            ],
        )

        demo.load(
            fn=lambda: None,
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
