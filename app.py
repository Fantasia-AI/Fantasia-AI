import gradio as gr
from huggingface_hub import InferenceClient
import os
import pandas as pd
from typing import List, Tuple

# LLM 모델 정의
LLM_MODELS = {
    "Default": "CohereForAI/c4ai-command-r-plus-08-2024",  # 기본 모델
    "Meta": "meta-llama/Llama-3.3-70B-Instruct",    
    "Mistral": "mistralai/Mistral-Nemo-Instruct-2407",
    "Alibaba": "Qwen/QwQ-32B-Preview"
}

def get_client(model_name):
    return InferenceClient(LLM_MODELS[model_name], token=os.getenv("HF_TOKEN"))

def analyze_file_content(content, file_type):
    """파일 내용을 분석하여 1줄 요약을 반환"""
    if file_type == 'parquet':
        return f"데이터셋 분석: {content.count('|')-1}개 컬럼의 데이터 테이블"
    
    # 텍스트 파일의 경우
    lines = content.split('\n')
    total_lines = len(lines)
    non_empty_lines = len([line for line in lines if line.strip()])
    
    if 'def ' in content or 'class ' in content:
        functions = len([line for line in lines if 'def ' in line])
        classes = len([line for line in lines if 'class ' in line])
        return f"코드 분석: {total_lines}줄의 Python 코드 ({functions}개 함수, {classes}개 클래스 포함)"
    else:
        return f"텍스트 분석: {total_lines}줄의 텍스트 문서 (유효 내용 {non_empty_lines}줄)"

def read_uploaded_file(file):
    if file is None:
        return "", ""
    try:
        if file.name.endswith('.parquet'):
            df = pd.read_parquet(file.name, engine='pyarrow')
            content = df.head(10).to_markdown(index=False)
            return content, "parquet"
        else:
            content = file.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            return content, "text"
    except Exception as e:
        return f"파일을 읽는 중 오류가 발생했습니다: {str(e)}", "error"

def format_history(history):
    formatted_history = []
    for user_msg, assistant_msg in history:
        formatted_history.append({"role": "user", "content": user_msg})
        if assistant_msg:
            formatted_history.append({"role": "assistant", "content": assistant_msg})
    return formatted_history

def chat(message, history, uploaded_file, model_name, system_message="", max_tokens=4000, temperature=0.7, top_p=0.9):
    system_prefix = """반드시 한글로 답변할것. 너는 주어진 소스코드나 데이터를 기반으로 "서비스 사용 설명 및 안내, Q&A를 하는 역할이다". 아주 친절하고 자세하게 4000토큰 이상 Markdown 형식으로 작성하라. 너는 입력된 내용을 기반으로 사용 설명 및 질의 응답을 진행하며, 이용자에게 도움을 주어야 한다."""

    if uploaded_file:
        content, file_type = read_uploaded_file(uploaded_file)
        if file_type == "error":
            return "", history + [[message, content]]
        
        # 파일 내용 분석 및 요약
        file_summary = analyze_file_content(content, file_type)
        
        if file_type == 'parquet':
            system_message += f"\n\n파일 내용:\n```markdown\n{content}\n```"
        else:
            system_message += f"\n\n파일 내용:\n```python\n{content}\n```"
            
        if message == "파일 분석을 시작합니다.":
            message = f"""[파일 요약] {file_summary}

다음 내용을 포함하여 상세히 설명하라:
1. 파일의 주요 목적과 기능
2. 주요 특징과 구성요소
3. 활용 방법 및 사용 시나리오
4. 주의사항 및 제한사항
5. 기대효과 및 장점"""

    messages = [{"role": "system", "content": f"{system_prefix} {system_message}"}]
    messages.extend(format_history(history))
    messages.append({"role": "user", "content": message})

    response = ""
    try:
        client = get_client(model_name)
        for msg in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            token = msg.choices[0].delta.get('content', None)
            if token:
                response += token
        
        history = history + [[message, response]]
        return "", history
    except Exception as e:
        error_msg = f"추론 중 오류가 발생했습니다: {str(e)}"
        history = history + [[message, error_msg]]
        return "", history

css = """
footer {visibility: hidden}
"""

with gr.Blocks(theme="Yntec/HaleyCH_Theme_Orange", css=css) as demo:
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=600)
            msg = gr.Textbox(
                label="메시지를 입력하세요",
                show_label=False,
                placeholder="메시지를 입력하세요...",
                container=False
            )
            clear = gr.ClearButton([msg, chatbot])
        
        with gr.Column(scale=1):
            model_name = gr.Radio(
                choices=list(LLM_MODELS.keys()),
                value="Default",
                label="LLM 모델 선택",
                info="사용할 LLM 모델을 선택하세요"
            )
            
            file_upload = gr.File(
                label="파일 업로드",
                file_types=["text", ".parquet"],
                type="filepath"
            )
            
            with gr.Accordion("고급 설정", open=False):
                system_message = gr.Textbox(label="System Message", value="")
                max_tokens = gr.Slider(minimum=1, maximum=8000, value=4000, label="Max Tokens")
                temperature = gr.Slider(minimum=0, maximum=1, value=0.7, label="Temperature")
                top_p = gr.Slider(minimum=0, maximum=1, value=0.9, label="Top P")

    # 이벤트 바인딩
    msg.submit(
        chat,
        inputs=[msg, chatbot, file_upload, model_name, system_message, max_tokens, temperature, top_p],
        outputs=[msg, chatbot]
    )

    # 파일 업로드 시 자동 분석
    file_upload.change(
        chat,
        inputs=[gr.Textbox(value="파일 분석을 시작합니다."), chatbot, file_upload, model_name, system_message, max_tokens, temperature, top_p],
        outputs=[msg, chatbot]
    )

    # 예제 추가
    gr.Examples(
        examples=[
            ["상세한 사용 방법을 마치 화면을 보면서 설명하듯이 4000 토큰 이상 자세히 설명하라"],
            ["FAQ 20건을 상세하게 작성하라. 4000토큰 이상 사용하라."],
            ["사용 방법과 차별점, 특징, 강점을 중심으로 4000 토큰 이상 유튜브 영상 스크립트 형태로 작성하라"],
            ["본 서비스를 SEO 최적화하여 블로그 포스트로 4000 토큰 이상 작성하라"],
            ["특허 출원에 활용할 기술 및 비즈니스모델 측면을 포함하여 특허 출원서 구성에 맞게 작성하라"],
            ["계속 이어서 답변하라"],
        ],
        inputs=msg,
    )

if __name__ == "__main__":
    demo.launch()