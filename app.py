import gradio as gr
from transformers import pipeline
import os
from typing import List, Tuple, Generator
import concurrent.futures

# Hugging Face 토큰 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 경고 메시지 방지
HF_TOKEN = os.getenv("HF_TOKEN")

# Available LLM models
LLM_MODELS = {
    "Llama-3.3": "meta-llama/Llama-3.3-70B-Instruct",
    "QwQ-32B": "Qwen/QwQ-32B-Preview",
    "C4AI-Command": "CohereForAI/c4ai-command-r-plus-08-2024",
    "Marco-o1": "AIDC-AI/Marco-o1",
    "Qwen2.5": "Qwen/Qwen2.5-72B-Instruct",
    "Mistral-Nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    "Nemotron-70B": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
}

# Default selected models
DEFAULT_MODELS = [
    "meta-llama/Llama-3.3-70B-Instruct",
    "CohereForAI/c4ai-command-r-plus-08-2024",
    "mistralai/Mistral-Nemo-Instruct-2407"
]

# Pipeline 초기화
pipes = {}
for model_name in LLM_MODELS.values():
    try:
        pipes[model_name] = pipeline(
            "text-generation",
            model=model_name,
            token=HF_TOKEN,
            device_map="auto"
        )
    except Exception as e:
        print(f"Failed to load model {model_name}: {str(e)}")

def process_file(file) -> str:
    if file is None:
        return ""
    if file.name.endswith(('.txt', '.md')):
        return file.read().decode('utf-8')
    return f"Uploaded file: {file.name}"

def format_messages(message: str, history: List[Tuple[str, str]], system_message: str) -> List[dict]:
    messages = [{"role": "system", "content": system_message}]
    
    for user, assistant in history:
        if user:
            messages.append({"role": "user", "content": user})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
    
    messages.append({"role": "user", "content": message})
    return messages

def generate_response(
    pipe,
    messages: List[dict],
    max_tokens: int,
    temperature: float,
    top_p: float
) -> Generator[str, None, None]:
    try:
        formatted_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        response = pipe(
            formatted_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=50256,
            num_return_sequences=1,
            streaming=True
        )
        
        generated_text = ""
        for output in response:
            new_text = output[0]['generated_text'][len(formatted_prompt):].strip()
            generated_text = new_text
            yield generated_text
            
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
) -> Tuple[Generator[str, None, None], Generator[str, None, None], Generator[str, None, None]]:
    if file:
        file_content = process_file(file)
        message = f"{message}\n\nFile content:\n{file_content}"

    while len(selected_models) < 3:
        selected_models.append(selected_models[-1])

    def generate(pipe, history):
        messages = format_messages(message, history, system_message)
        return generate_response(pipe, messages, max_tokens, temperature, top_p)

    return (
        generate(pipes[selected_models[0]], history1),
        generate(pipes[selected_models[1]], history2),
        generate(pipes[selected_models[2]], history3),
    )

with gr.Blocks() as demo:
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
                value="You are a friendly Chatbot.",
                label="System message"
            )
            max_tokens = gr.Slider(
                minimum=1,
                maximum=2048,
                value=512,
                step=1,
                label="Max new tokens"
            )
            temperature = gr.Slider(
                minimum=0.1,
                maximum=4.0,
                value=0.7,
                step=0.1,
                label="Temperature"
            )
            top_p = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.95,
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
    # Hugging Face 토큰이 설정되어 있는지 확인
    if not HF_TOKEN:
        print("Warning: HF_TOKEN environment variable is not set")
    demo.launch()