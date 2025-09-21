from __future__ import annotations

import gradio as gr

from Pharmagent.app.configurations import ClientRuntimeConfig
from Pharmagent.app.constants import (
    AGENT_MODEL_CHOICES,
    CLOUD_PROVIDERS,
    PARSING_MODEL_CHOICES,
)
from Pharmagent.app.client.controllers import (
    reset_agent_fields,
    run_agent,
    set_agent_model,
    set_llm_provider,
    set_parsing_model,
    toggle_cloud_services,
)


###############################################################################
def create_interface() -> gr.Blocks:
    with gr.Blocks(
        title="Pharmagent Clinical Copilot",
        analytics_enabled=False,
        theme="soft",
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
                process_from_files = gr.Checkbox(
                    label="Process patients from files",
                    value=False,
                )
                with gr.Column():
                    run_button = gr.Button("Run Workflow", variant="primary")
                    rag_button = gr.Button("Load RAG documents", variant="secondary")
                    clear_button = gr.Button("Clear all")
                output = gr.Textbox(
                    label="Agent Output",
                    lines=30,
                    show_copy_button=True,
                    interactive=False,
                )

        with gr.Accordion("Runtime Configuration", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    use_cloud_services = gr.Checkbox(
                        label="Use Cloud Services",
                        value=ClientRuntimeConfig.is_cloud_enabled(),
                    )
                    llm_provider_dropdown = gr.Dropdown(
                        label="LLM Provider",
                        choices=CLOUD_PROVIDERS,
                        value=ClientRuntimeConfig.get_llm_provider(),
                        interactive=ClientRuntimeConfig.is_cloud_enabled(),
                    )
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
                with gr.Column(scale=1):
                    placeholder_button = gr.Button(
                        "Configure Provider",
                        variant="secondary",
                    )

        use_cloud_services.change(
            fn=toggle_cloud_services,
            inputs=use_cloud_services,
            outputs=llm_provider_dropdown,
        )
        llm_provider_dropdown.change(
            fn=set_llm_provider,
            inputs=llm_provider_dropdown,
            outputs=llm_provider_dropdown,
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

        run_button.click(
            fn=run_agent,
            inputs=[
                patient_name,
                anamnesis,
                drugs,
                exams,
                alt,
                alt_max,
                alp,
                alp_max,
                symptoms,
                process_from_files,
            ],
            outputs=output,
            api_name="run_agent",
        )
        clear_button.click(
            fn=reset_agent_fields,
            outputs=[
                patient_name,
                anamnesis,
                drugs,
                exams,
                alt,
                alt_max,
                alp,
                alp_max,
                symptoms,
                process_from_files,
                output,
                use_cloud_services,
                llm_provider_dropdown,
                parsing_model_dropdown,
                agent_model_dropdown,
            ],
        )
        clear_button.click(lambda: "", outputs=output)

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
