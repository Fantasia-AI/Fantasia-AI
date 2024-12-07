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
    """파일 내용을 분석하여 구조적 요약을 반환"""
    if file_type in ['parquet', 'csv']:
        try:
            # 데이터셋 구조 분석
            lines = content.split('\n')
            header = lines[0]
            columns = header.count('|') - 1
            rows = len(lines) - 3  # 헤더와 구분선 제외
            return f"데이터셋 구조: {columns}개 컬럼, {rows}개 데이터 샘플"
        except:
            return "데이터셋 구조 분석 실패"
    
    # 텍스트/코드 파일의 경우
    lines = content.split('\n')
    total_lines = len(lines)
    non_empty_lines = len([line for line in lines if line.strip()])
    
    if any(keyword in content.lower() for keyword in ['def ', 'class ', 'import ', 'function']):
        functions = len([line for line in lines if 'def ' in line])
        classes = len([line for line in lines if 'class ' in line])
        imports = len([line for line in lines if 'import ' in line or 'from ' in line])
        return f"코드 구조 분석: 총 {total_lines}줄 (함수 {functions}개, 클래스 {classes}개, 임포트 {imports}개)"
    
    paragraphs = content.count('\n\n') + 1
    words = len(content.split())
    return f"문서 구조 분석: 총 {total_lines}줄, {paragraphs}개 문단, 약 {words}개 단어"

def read_uploaded_file(file):
    if file is None:
        return "", ""
    try:
        file_ext = os.path.splitext(file.name)[1].lower()
        
        if file_ext == '.parquet':
            df = pd.read_parquet(file.name, engine='pyarrow')
            content = df.head(10).to_markdown(index=False)
            return content, "parquet"
        elif file_ext == '.csv':
            df = pd.read_csv(file.name)
            content = f"데이터 미리보기:\n{df.head(10).to_markdown(index=False)}\n\n"
            content += f"\n데이터 정보:\n"
            content += f"- 총 행 수: {len(df)}\n"
            content += f"- 총 열 수: {len(df.columns)}\n"
            content += f"- 컬럼 목록: {', '.join(df.columns)}\n"
            return content, "csv"
        else:
            with open(file.name, 'r', encoding='utf-8') as f:
                content = f.read()
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
    system_prefix = """너는 파일 분석 전문가입니다. 업로드된 파일의 내용을 깊이 있게 분석하여 다음과 같은 관점에서 설명해야 합니다:

1. 파일의 전반적인 구조와 구성
2. 주요 내용과 패턴 분석
3. 데이터의 특징과 의미
   - 데이터셋의 경우: 컬럼의 의미, 데이터 타입, 값의 분포
   - 텍스트/코드의 경우: 구조적 특징, 주요 패턴
4. 잠재적 활용 방안
5. 데이터 품질 및 개선 가능한 부분

전문가적 관점에서 상세하고 구조적인 분석을 제공하되, 이해하기 쉽게 설명하세요. 분석 결과는 Markdown 형식으로 작성하고, 가능한 한 구체적인 예시를 포함하세요."""

    if uploaded_file:
        content, file_type = read_uploaded_file(uploaded_file)
        if file_type == "error":
            yield "", history + [[message, content]]
            return
        
        # 파일 내용 분석 및 구조적 요약
        file_summary = analyze_file_content(content, file_type)
        
        if file_type in ['parquet', 'csv']:
            system_message += f"\n\n파일 내용:\n```markdown\n{content}\n```"
        else:
            system_message += f"\n\n파일 내용:\n```\n{content}\n```"
            
        if message == "파일 분석을 시작합니다.":
            message = f"""[구조 분석] {file_summary}

다음 관점에서 상세 분석을 제공해주세요:
1. 파일의 전반적인 구조와 형식
2. 주요 내용 및 구성요소 분석
3. 데이터/내용의 특징과 패턴
4. 품질 및 완성도 평가
5. 개선 가능한 부분 제안
6. 실제 활용 방안 및 추천사항"""

    messages = [{"role": "system", "content": f"{system_prefix} {system_message}"}]
    messages.extend(format_history(history))
    messages.append({"role": "user", "content": message})

    try:
        client = get_client(model_name)
        partial_message = ""
        
        for msg in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            token = msg.choices[0].delta.get('content', None)
            if token:
                partial_message += token
                yield "", history + [[message, partial_message]]
                
    except Exception as e:
        error_msg = f"추론 중 오류가 발생했습니다: {str(e)}"
        yield "", history + [[message, error_msg]]

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
                label="파일 업로드 (텍스트, 코드, CSV, Parquet 파일)",
                file_types=["text", ".csv", ".parquet"],
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
        outputs=[msg, chatbot],
        queue=True
    ).then(
        lambda: gr.update(interactive=True),
        None,
        [msg]
    )

    # 파일 업로드 시 자동 분석
    file_upload.change(
        chat,
        inputs=[gr.Textbox(value="파일 분석을 시작합니다."), chatbot, file_upload, model_name, system_message, max_tokens, temperature, top_p],
        outputs=[msg, chatbot],
        queue=True
    )

    # 예제 추가
    gr.Examples(
        examples=[
            ["파일의 전반적인 구조와 특징을 자세히 설명해주세요."],
            ["이 파일의 주요 패턴과 특징을 분석해주세요."],
            ["파일의 품질과 개선 가능한 부분을 평가해주세요."],
            ["이 파일을 실제로 어떻게 활용할 수 있을까요?"],
            ["파일의 주요 내용을 요약하고 핵심 인사이트를 도출해주세요."],
            ["이전 분석을 이어서 더 자세히 설명해주세요."],
        ],
        inputs=msg,
    )

if __name__ == "__main__":
    demo.launch()