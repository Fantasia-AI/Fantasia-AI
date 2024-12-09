import os
from dotenv import load_dotenv
import gradio as gr
from huggingface_hub import InferenceClient
import pandas as pd
from typing import List, Tuple
import json
from datetime import datetime

# 환경 변수 설정
HF_TOKEN = os.getenv("HF_TOKEN")

# LLM Models Definition
LLM_MODELS = {
    "Cohere c4ai-crp-08-2024": "CohereForAI/c4ai-command-r-plus-08-2024",  # Default
    "Meta Llama3.3-70B": "meta-llama/Llama-3.3-70B-Instruct"    # Backup model
}

# 대화 히스토리를 저장할 클래스
class ChatHistory:
    def __init__(self):
        self.history = []
        self.history_file = "/tmp/chat_history.json"
        self.load_history()

    def add_conversation(self, user_msg: str, assistant_msg: str):
        conversation = {
            "timestamp": datetime.now().isoformat(),
            "conversation": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ]
        }
        self.history.append(conversation)
        self.save_history()

    def get_recent_conversations(self, limit=10):
        return self.history[-limit:] if self.history else []

    def format_for_display(self):
        formatted = []
        for conv in self.history:
            formatted.extend([
                [conv["conversation"][0]["content"], conv["conversation"][1]["content"]]
            ])
        return formatted

    def clear_history(self):
        self.history = []
        self.save_history()

    def save_history(self):
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"히스토리 저장 실패: {e}")

    def load_history(self):
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
        except Exception as e:
            print(f"히스토리 로드 실패: {e}")
            self.history = []

# 전역 ChatHistory 인스턴스 생성
chat_history = ChatHistory()

def get_client(model_name="Cohere c4ai-crp-08-2024"):
    try:
        return InferenceClient(LLM_MODELS[model_name], token=HF_TOKEN)
    except Exception:
        return InferenceClient(LLM_MODELS["Meta Llama3.3-70B"], token=HF_TOKEN)

def analyze_file_content(content, file_type):
    """Analyze file content and return structural summary"""
    if file_type in ['parquet', 'csv']:
        try:
            lines = content.split('\n')
            header = lines[0]
            columns = header.count('|') - 1
            rows = len(lines) - 3
            return f"📊 데이터셋 구조: {columns}개 컬럼, {rows}개 데이터"
        except:
            return "❌ 데이터셋 구조 분석 실패"
    
    lines = content.split('\n')
    total_lines = len(lines)
    non_empty_lines = len([line for line in lines if line.strip()])
    
    if any(keyword in content.lower() for keyword in ['def ', 'class ', 'import ', 'function']):
        functions = len([line for line in lines if 'def ' in line])
        classes = len([line for line in lines if 'class ' in line])
        imports = len([line for line in lines if 'import ' in line or 'from ' in line])
        return f"💻 코드 구조: {total_lines}줄 (함수: {functions}, 클래스: {classes}, 임포트: {imports})"
    
    paragraphs = content.count('\n\n') + 1
    words = len(content.split())
    return f"📝 문서 구조: {total_lines}줄, {paragraphs}단락, 약 {words}단어"

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
            encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1']
            for encoding in encodings:
                try:
                    df = pd.read_csv(file.name, encoding=encoding)
                    content = f"📊 데이터 미리보기:\n{df.head(10).to_markdown(index=False)}\n\n"
                    content += f"\n📈 데이터 정보:\n"
                    content += f"- 전체 행 수: {len(df)}\n"
                    content += f"- 전체 열 수: {len(df.columns)}\n"
                    content += f"- 컬럼 목록: {', '.join(df.columns)}\n"
                    content += f"\n📋 컬럼 데이터 타입:\n"
                    for col, dtype in df.dtypes.items():
                        content += f"- {col}: {dtype}\n"
                    null_counts = df.isnull().sum()
                    if null_counts.any():
                        content += f"\n⚠️ 결측치:\n"
                        for col, null_count in null_counts[null_counts > 0].items():
                            content += f"- {col}: {null_count}개 누락\n"
                    return content, "csv"
                except UnicodeDecodeError:
                    continue
            raise UnicodeDecodeError(f"❌ 지원되는 인코딩으로 파일을 읽을 수 없습니다 ({', '.join(encodings)})")
        else:
            encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1']
            for encoding in encodings:
                try:
                    with open(file.name, 'r', encoding=encoding) as f:
                        content = f.read()
                    return content, "text"
                except UnicodeDecodeError:
                    continue
            raise UnicodeDecodeError(f"❌ 지원되는 인코딩으로 파일을 읽을 수 없습니다 ({', '.join(encodings)})")
    except Exception as e:
        return f"❌ 파일 읽기 오류: {str(e)}", "error"


def chat(message, history, uploaded_file, system_message="", max_tokens=4000, temperature=0.7, top_p=0.9):
    if not message:
        return "", history

    system_prefix = """저는 여러분의 친근하고 지적인 AI 어시스턴트 'GiniGEN'입니다.. 다음과 같은 원칙으로 소통하겠습니다:

1. 🤝 친근하고 공감적인 태도로 대화
2. 💡 명확하고 이해하기 쉬운 설명 제공
3. 🎯 질문의 의도를 정확히 파악하여 맞춤형 답변
4. 📚 필요한 경우 업로드된 파일 내용을 참고하여 구체적인 도움 제공
5. ✨ 추가적인 통찰과 제안을 통한 가치 있는 대화

항상 예의 바르고 친절하게 응답하며, 필요한 경우 구체적인 예시나 설명을 추가하여 
이해를 돕겠습니다."""

    try:
        if uploaded_file:
            content, file_type = read_uploaded_file(uploaded_file)
            if file_type == "error":
                error_message = content
                chat_history.add_conversation(message, error_message)
                return "", history + [[message, error_message]]
            
            file_summary = analyze_file_content(content, file_type)
            
            if file_type in ['parquet', 'csv']:
                system_message += f"\n\n파일 내용:\n```markdown\n{content}\n```"
            else:
                system_message += f"\n\n파일 내용:\n```\n{content}\n```"
                
            if message == "파일 분석을 시작합니다...":
                message = f"""[파일 구조 분석] {file_summary}

다음 관점에서 도움을 드리겠습니다:
1. 📋 전반적인 내용 파악
2. 💡 주요 특징 설명
3. 🎯 실용적인 활용 방안
4. ✨ 개선 제안
5. 💬 추가 질문이나 필요한 설명"""

        # 시스템 메시지 및 히스토리 설정
        messages = [{"role": "system", "content": system_prefix + system_message}]
        
        # 이전 대화 히스토리 추가
        if history:
            for h in history:
                messages.append({"role": "user", "content": h[0]})
                if h[1]:
                    messages.append({"role": "assistant", "content": h[1]})
        
        messages.append({"role": "user", "content": message})

        client = get_client()
        partial_message = ""
        
        # 스트리밍 응답 처리
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
                current_history = history + [[message, partial_message]]
                yield "", current_history

        # 완성된 대화 저장
        chat_history.add_conversation(message, partial_message)
        
    except Exception as e:
        error_msg = f"❌ 오류가 발생했습니다: {str(e)}"
        chat_history.add_conversation(message, error_msg)
        yield "", history + [[message, error_msg]]


with gr.Blocks(theme="Yntec/HaleyCH_Theme_Orange", title="GiniGEN 🤖") as demo:
    # 기존 히스토리 로드
    initial_history = chat_history.format_for_display()
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                value=initial_history,  # 초기 히스토리 설정
                height=600, 
                label="대화창 💬",
                show_label=True,
                type="messages"
            )
            msg = gr.Textbox(
                label="메시지 입력",
                show_label=False,
                placeholder="무엇이든 물어보세요... 💭",
                container=False
            )
            with gr.Row():
                clear = gr.ClearButton([msg, chatbot], value="대화내용 지우기")
                send = gr.Button("보내기 📤")
        
        with gr.Column(scale=1):
            gr.Markdown("### GiniGEN 🤖 [파일 업로드] 📁\n지원 형식: 텍스트, 코드, CSV, Parquet 파일")
            file_upload = gr.File(
                label="파일 선택",
                file_types=["text", ".csv", ".parquet"],
                type="filepath"
            )
            
            with gr.Accordion("고급 설정 ⚙️", open=False):
                system_message = gr.Textbox(label="시스템 메시지 📝", value="")
                max_tokens = gr.Slider(minimum=1, maximum=8000, value=4000, label="최대 토큰 수 📊")
                temperature = gr.Slider(minimum=0, maximum=1, value=0.7, label="창의성 수준 🌡️")
                top_p = gr.Slider(minimum=0, maximum=1, value=0.9, label="응답 다양성 📈")

    # 예시 질문
    gr.Examples(
        examples=[
            ["안녕하세요! 어떤 도움이 필요하신가요? 🤝"],
            ["제가 이해하기 쉽게 설명해 주시겠어요? 📚"],
            ["이 내용을 실제로 어떻게 활용할 수 있을까요? 🎯"],
            ["추가로 조언해 주실 내용이 있으신가요? ✨"],
            ["궁금한 점이 더 있는데 여쭤봐도 될까요? 🤔"],
        ],
        inputs=msg,
    )

    # 대화내용 지우기 버튼에 히스토리 초기화 기능 추가
    def clear_chat():
        chat_history.clear_history()
        return None, None

    # 이벤트 바인딩
    msg.submit(
        chat,
        inputs=[msg, chatbot, file_upload, system_message, max_tokens, temperature, top_p],
        outputs=[msg, chatbot]
    )

    send.click(
        chat,
        inputs=[msg, chatbot, file_upload, system_message, max_tokens, temperature, top_p],
        outputs=[msg, chatbot]
    )

    clear.click(
        clear_chat,
        outputs=[msg, chatbot]
    )

    # 파일 업로드시 자동 분석
    file_upload.change(
        lambda: "파일 분석을 시작합니다...",
        outputs=msg
    ).then(
        chat,
        inputs=[msg, chatbot, file_upload, system_message, max_tokens, temperature, top_p],
        outputs=[msg, chatbot]
    )

if __name__ == "__main__":
    demo.launch()        