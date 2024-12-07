import gradio as gr
from huggingface_hub import InferenceClient
import os
from typing import List, Tuple

# Hugging Face 토큰 설정
HF_TOKEN = os.getenv("HF_TOKEN")

# Available LLM models
LLM_MODELS = {
    "Mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "Zephyr": "HuggingFaceH4/zephyr-7b-beta",
    "OpenChat": "openchat/openchat-3.5",
    "Llama2": "meta-llama/Llama-2-7b-chat-hf",
    "Phi": "microsoft/phi-2",
    "Neural": "nvidia/neural-chat-7b-v3-1",
    "Starling": "HuggingFaceH4/starling-lm-7b-alpha"
}

# Default selected models
DEFAULT_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "HuggingFaceH4/zephyr-7b-beta",
    "openchat/openchat-3.5"
]

# Initialize clients with token
clients = {
    model: InferenceClient(model, token=HF_TOKEN) 
    for model in LLM_MODELS.values()
}

def process_file(file) -> str:
    if file is None:
        return ""
    if file.name.endswith(('.txt', '.md')):
        return file.read().decode('utf-8')
    return f"Uploaded file: {file.name}"

def respond_single(
    client,
    message: str,
    history: List[Tuple[str, str]],
    system_message: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
):
    system_prefix = """반드시 한글로 답변할것. 너는 주어진 내용을 기반으로 상세한 설명과 Q&A를 제공하는 역할이다. 
    아주 친절하고 자세하게 설명하라."""
    
    messages = [{"role": "system", "content": f"{system_prefix} {system_message}"}]
    
    for user, assistant in history:
        if user:
            messages.append({"role": "user", "content": user})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
    
    messages.append({"role": "user", "content": message})
    
    response = ""
    try:
        for msg in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            if hasattr(msg.choices[0].delta, 'content'):
                token = msg.choices[0].delta.content
                if token is not None:
                    response += token
                    yield response
    except Exception as e:
        yield f"Error: {str(e)}"

def respond_all(
    message: str,
    file,
    history1: List[Tuple[str, str]],
    history2: List[Tuple[str, str]],
    history3: List[Tuple[str, str]],
    selected_models: List[str],
    system_message: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
):
    if file:
        file_content = process_file(file)
        message = f"{message}\n\nFile content:\n{file_content}"

    while len(selected_models) < 3:
        selected_models.append(selected_models[-1])

    def generate(client, history):
        return respond_single(
            client,
            message,
            history,
            system_message,
            max_tokens,
            temperature,
            top_p,
        )

    return (
        generate(clients[selected_models[0]], history1),
        generate(clients[selected_models[1]], history2),
        generate(clients[selected_models[2]], history3),
    )

css = """
footer {visibility: hidden}
"""

with gr.Blocks(theme="Nymbo/Nymbo_Theme", css=css) as demo:
    with gr.Row():
        model_choices = gr.Checkboxgroup(
            choices=list(LLM_MODELS.values()),
            value=DEFAULT_MODELS,
            label="Select Models (Choose up to 3)",
            interactive=True
        )

    with gr.Row():
        with gr.Column():
            chat1 = gr.ChatInterface(
                lambda message, history: None,
                chatbot=gr.Chatbot(height=400, label="Chat 1"),
                textbox=False,
            )
        with gr.Column():
            chat2 = gr.ChatInterface(
                lambda message, history: None,
                chatbot=gr.Chatbot(height=400, label="Chat 2"),
                textbox=False,
            )
        with gr.Column():
            chat3 = gr.ChatInterface(
                lambda message, history: None,
                chatbot=gr.Chatbot(height=400, label="Chat 3"),
                textbox=False,
            )

    with gr.Row():
        with gr.Column():
            system_message = gr.Textbox(
                value="당신은 친절한 AI 어시스턴트입니다.",
                label="System message"
            )
            max_tokens = gr.Slider(
                minimum=1,
                maximum=8000,
                value=4000,
                step=1,
                label="Max new tokens"
            )
            temperature = gr.Slider(
                minimum=0,
                maximum=1,
                value=0.7,
                step=0.1,
                label="Temperature"
            )
            top_p = gr.Slider(
                minimum=0,
                maximum=1,
                value=0.9,
                step=0.05,
                label="Top-p"
            )
            
    with gr.Row():
        file_input = gr.File(label="Upload File (optional)")
        msg_input = gr.Textbox(
            show_label=False,
            placeholder="Enter text and press enter",
            container=False
        )

    examples = [
        ["상세한 사용 방법을 마치 화면을 보면서 설명하듯이 4000 토큰 이상 자세히 설명하라"],
        ["FAQ 20건을 상세하게 작성하라. 4000토큰 이상 사용하라."],
        ["사용 방법과 차별점, 특징, 강점을 중심으로 4000 토큰 이상 유튜브 영상 스크립트 형태로 작성하라"],
        ["본 서비스를 SEO 최적화하여 블로그 포스트로 4000 토큰 이상 작성하라"],
        ["계속 이어서 답변하라"],
    ]
    
    gr.Examples(
        examples=examples,
        inputs=msg_input,
        cache_examples=False
    )
        
    def submit_message(message, file):
        return respond_all(
            message,
            file,
            chat1.chatbot.value,
            chat2.chatbot.value,
            chat3.chatbot.value,
            model_choices.value,
            system_message.value,
            max_tokens.value,
            temperature.value,
            top_p.value,
        )

    msg_input.submit(
        submit_message,
        [msg_input, file_input],
        [chat1.chatbot, chat2.chatbot, chat3.chatbot],
        api_name="submit"
    )

if __name__ == "__main__":
    if not HF_TOKEN:
        print("Warning: HF_TOKEN environment variable is not set")
    demo.launch()