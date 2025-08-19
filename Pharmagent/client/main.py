import gradio as gr
from Pharmagent.client.controllers import run_agent

with gr.Blocks(title="Patient Agent UI", analytics_enabled=False) as demo:
    gr.Markdown("### Patient Agent")

    name = gr.Textbox(
        label="Patient name",
        placeholder="e.g., Marco Rossi",
        lines=1,
    )

    input_box = gr.Textbox(
        label="Input text",
        placeholder="Paste or type notes / instructions for the agent...",
        lines=10,
    )

    run_btn = gr.Button("Start Agent", variant="primary")

    output_box = gr.Textbox(
        label="Agent output",
        lines=12,
        show_copy_button=True,
        interactive=False,
    )

    run_btn.click(fn=run_agent, inputs=[name, input_box], outputs=output_box, api_name="run_agent")
    input_box.change(lambda _: "", inputs=input_box, outputs=output_box)

if __name__ == "__main__":
    demo.queue(max_size=32).launch(
        server_name="127.0.0.1",
        server_port=7861,
        inbrowser=True  
    )