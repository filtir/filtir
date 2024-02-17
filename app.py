# gradio_app.py

import gradio as gr
from run_pipeline import get_fact_checked


def fact_check_function(text, model):
    # Assume the text is already read from the user input, so we don't need to open a file here
    out = get_fact_checked(text, mode="fast", model=model)
    return out["fact_checked_md"]


def create_gradio_interface():
    iface = gr.Interface(
        title="Filtir - Fact-Checking AI generated content",
        allow_flagging=False,
        fn=fact_check_function,
        inputs=[
            gr.Textbox(
                lines=10, placeholder="Enter text to fact-check...", label="Input Text"
            ),
            gr.Dropdown(choices=["gpt-4-1106-preview"], label="Model"),
        ],
        outputs=gr.Markdown(label="Filtir Output"),
    )
    return iface


if __name__ == "__main__":
    iface = create_gradio_interface()
    iface.launch()
