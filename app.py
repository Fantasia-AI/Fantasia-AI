

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

# Gradio 인터페이스 설정
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.File(label="Fashion Code File", file_types=[".cod", ".txt", ".py"]),
        gr.File(label="UHD Image Code File", file_types=[".cod", ".txt", ".py"]),
        gr.File(label="MixGEN Code File", file_types=[".cod", ".txt", ".py"]),
        gr.File(label="Parquet File", file_types=[".parquet"]),
        gr.Textbox(label="System Message", value=""),
        gr.Slider(minimum=1, maximum=8000, value=4000, label="Max Tokens"),
        gr.Slider(minimum=0, maximum=1, value=0.7, label="Temperature"),
        gr.Slider(minimum=0, maximum=1, value=0.9, label="Top P"),
    ],
    examples=[
        ["패션 코드 실행"],
        ["UHD 이미지 코드 실행"],
        ["MixGEN 코드 실행"],
        ["test.parquet 실행"],
        ["상세한 사용 방법을 마치 화면을 보면서 설명하듯이 4000 토큰 이상 자세히 설명하라"],
        ["FAQ 20건을 상세하게 작성하라. 4000토큰 이상 사용하라."],
        ["사용 방법과 차별점, 특징, 강점을 중심으로 4000 토큰 이상 유튜브 영상 스크립트 형태로 작성하라"],
        ["본 서비스를 SEO 최적화하여 블로그 포스트로 4000 토큰 이상 작성하라"],
        ["특허 출원에 활용할 기술 및 비즈니스모델 측면을 포함하여 특허 출원서 구성에 맞게 작성하라"],
        ["계속 이어서 답변하라"],
    ],
    theme="Nymbo/Nymbo_Theme",
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
