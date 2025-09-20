from __future__ import annotations

import gradio as gr

from Pharmagent.app.client.controllers import reset_agent_fields, run_agent


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
                    lines=8,
                    max_lines=16,
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
                    lines=8,
                    max_lines=16,
                )
                exams = gr.Textbox(
                    label="Additional Exams",
                    placeholder="Provide lab or imaging exam notes...",
                    lines=8,
                    max_lines=16,
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
                    reset_button = gr.Button("Reset Form", variant="secondary")
                    clear_button = gr.Button("Clear Output")
                output = gr.Textbox(
                    label="Agent Output",
                    lines=18,
                    show_copy_button=True,
                    interactive=False,
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
        reset_button.click(
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


