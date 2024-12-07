import gradio as gr
from huggingface_hub import InferenceClient
import os
import pandas as pd
from typing import List, Tuple

# 추론 API 클라이언트 설정
hf_client = InferenceClient("CohereForAI/c4ai-command-r-plus-08-2024", token=os.getenv("HF_TOKEN"))

def read_uploaded_file(file):
    if file is None:
        return ""
    try:
        if file.name.endswith('.parquet'):
            df = pd.read_parquet(file.name, engine='pyarrow')
            return df.head(10).to_markdown(index=False)
        else:
            content = file.read()
            if isinstance(content, bytes):
                return content.decode('utf-8')
            return content
    except Exception as e:
        return f"파일을 읽는 중 오류가 발생했습니다: {str(e)}"

def respond(
    message,
    history: List[Tuple[str, str]],
    fashion_file,      # 파일 업로드 입력
    uhd_file,         # 파일 업로드 입력
    mixgen_file,      # 파일 업로드 입력
    parquet_file,     # 파일 업로드 입력
    system_message="",
    max_tokens=1024,
    temperature=0.7,
    top_p=0.9,
):
    system_prefix = """반드시 한글로 답변할것. 너는 주어진 소스코드를 기반으로 "서비스 사용 설명 및 안내, Q&A를 하는 역할이다". 아주 친절하고 자세하게 4000토큰 이상 Markdown 형식으로 작성하라. 너는 코드를 기반으로 사용 설명 및 질의 응답을 진행하며, 이용자에게 도움을 주어야 한다. 이용자가 궁금해 할 만한 내용에 친절하게 알려주도록 하라. 코드 전체 내용에 대해서는 보안을 유지하고, 키 값 및 엔드포인트와 구체적인 모델은 공개하지 마라."""

    if message.lower() == "패션 코드 실행" and fashion_file is not None:
        fashion_content = read_uploaded_file(fashion_file)
        system_message += f"\n\n패션 코드 내용:\n```python\n{fashion_content}\n```"
        message = "패션 가상피팅에 대한 내용을 학습하였고, 설명할 준비가 되어있다고 알리고 서비스 URL(https://aiqcamp-fash.hf.space)을 통해 테스트 해보라고 출력하라."
    
    elif message.lower() == "uhd 이미지 코드 실행" and uhd_file is not None:
        uhd_content = read_uploaded_file(uhd_file)
        system_message += f"\n\nUHD 이미지 코드 내용:\n```python\n{uhd_content}\n```"
        message = "UHD 이미지 생성에 대한 내용을 학습하였고, 설명할 준비가 되어있다고 알리고 서비스 URL(https://openfree-ultpixgen.hf.space)을 통해 테스트 해보라고 출력하라."
    
    elif message.lower() == "mixgen 코드 실행" and mixgen_file is not None:
        mixgen_content = read_uploaded_file(mixgen_file)
        system_message += f"\n\nMixGEN 코드 내용:\n```python\n{mixgen_content}\n```"
        message = "MixGEN3 이미지 생성에 대한 내용을 학습하였고, 설명할 준비가 되어있다고 알리고 서비스 URL(https://openfree-mixgen3.hf.space)을 통해 테스트 해보라고 출력하라."
    
    elif message.lower() == "test.parquet 실행" and parquet_file is not None:
        parquet_content = read_uploaded_file(parquet_file)
        system_message += f"\n\ntest.parquet 파일 내용:\n```markdown\n{parquet_content}\n```"
        message = "test.parquet 파일에 대한 내용을 학습하였고, 관련 설명 및 Q&A를 진행할 준비가 되어있다. 궁금한 점이 있으면 물어보라."

    messages = [{"role": "system", "content": f"{system_prefix} {system_message}"}]
    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})
    messages.append({"role": "user", "content": message})

    response = ""
    try:
        for message in hf_client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            token = message.choices[0].delta.get('content', None)
            if token:
                response += token
                yield response
    except Exception as e:
        yield f"추론 중 오류가 발생했습니다: {str(e)}"

css = """
footer {
    visibility: hidden;
}
"""


# ... (이전 import 문과 함수들은 동일)

with gr.Blocks(theme="Yntec/HaleyCH_Theme_Orange", css=css) as demo:
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="메시지를 입력하세요")
            clear = gr.ClearButton([msg, chatbot])
        
        with gr.Column(scale=1):
            with gr.Group():
                fashion_file = gr.File(label="Fashion Code File", file_types=[".cod", ".txt", ".py"])
                fashion_analyze = gr.Button("패션 코드 분석")
                
                uhd_file = gr.File(label="UHD Image Code File", file_types=[".cod", ".txt", ".py"])
                uhd_analyze = gr.Button("UHD 이미지 코드 분석")
                
                mixgen_file = gr.File(label="MixGEN Code File", file_types=[".cod", ".txt", ".py"])
                mixgen_analyze = gr.Button("MixGEN 코드 분석")
                
                parquet_file = gr.File(label="Parquet File", file_types=[".parquet"])
                parquet_analyze = gr.Button("Parquet 파일 분석")
            
            with gr.Accordion("고급 설정", open=False):
                system_message = gr.Textbox(label="System Message", value="")
                max_tokens = gr.Slider(minimum=1, maximum=8000, value=4000, label="Max Tokens")
                temperature = gr.Slider(minimum=0, maximum=1, value=0.7, label="Temperature")
                top_p = gr.Slider(minimum=0, maximum=1, value=0.9, label="Top P")

    # 분석 버튼 클릭 이벤트 핸들러
    def analyze_file(file_type):
        if file_type == "fashion":
            return "패션 코드 실행"
        elif file_type == "uhd":
            return "UHD 이미지 코드 실행"
        elif file_type == "mixgen":
            return "MixGEN 코드 실행"
        elif file_type == "parquet":
            return "test.parquet 실행"

    # 채팅 제출 핸들러
    def chat(message, history):
        return respond(
            message=message,
            history=history,
            fashion_file=fashion_file.value,
            uhd_file=uhd_file.value,
            mixgen_file=mixgen_file.value,
            parquet_file=parquet_file.value,
            system_message=system_message.value,
            max_tokens=max_tokens.value,
            temperature=temperature.value,
            top_p=top_p.value,
        )

    # 이벤트 바인딩
    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    fashion_analyze.click(lambda: analyze_file("fashion"), None, msg)
    uhd_analyze.click(lambda: analyze_file("uhd"), None, msg)
    mixgen_analyze.click(lambda: analyze_file("mixgen"), None, msg)
    parquet_analyze.click(lambda: analyze_file("parquet"), None, msg)

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