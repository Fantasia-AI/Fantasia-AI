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

class ChatHistory:
    def __init__(self):
        self.history = []
        self.history_file = "/tmp/chat_history.json"
        self.load_history()

    def add_conversation(self, user_msg: str, assistant_msg: str):
        conversation = {
            "timestamp": datetime.now().isoformat(),
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ]
        }
        self.history.append(conversation)
        self.save_history()

    def format_for_display(self):
        # Gradio Chatbot 컴포넌트에 맞는 형식으로 변환
        formatted = []
        for conv in self.history:
            formatted.append([
                conv["messages"][0]["content"],  # user message
                conv["messages"][1]["content"]   # assistant message
            ])
        return formatted

    def get_messages_for_api(self):
        # API 호출을 위한 메시지 형식
        messages = []
        for conv in self.history:
            messages.extend([
                {"role": "user", "content": conv["messages"][0]["content"]},
                {"role": "assistant", "content": conv["messages"][1]["content"]}
            ])
        return messages

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

    system_prefix = """
You are 'FantasyAI✨', an advanced AI storyteller specialized in creating immersive fantasy narratives. Your purpose is to craft rich, detailed fantasy stories that incorporate classical and innovative elements of the genre. Your responses should start with 'FantasyAI✨:' and focus on creating engaging, imaginative content that bri시]"을 상황에 맞게 추가하여 소설 작성시 더욱 풍부하고 몰입감 있는 표현을 요청(출력)받은 언어로 표현하라.
[예시]
"고대의 마법이 깨어나며 대지가 울리는 소리가 들렸다..."
"용의 숨결이 하늘을 가르며, 구름을 불태웠다..."
"신비한 룬문자가 빛나며 공중에 떠올랐다..."
"엘프들의 노래가 숲을 울리자 나무들이 춤추기 시작했다..."
"예언의 말씀이 메아리치며 운명의 실이 움직이기 시작했다..."
"마법사의 지팡이에서 번쩍이는 빛이 어둠을 가르며..."
"고대 드워프의 대장간에서 전설의 검이 만들어지고 있었다..."
"수정구슬 속에 비친 미래의 환영이 서서히 모습을 드러냈다..."
"신성한 결계가 깨어지며 봉인된 악이 깨어났다..."
"영웅의 발걸음이 운명의 길을 따라 울려 퍼졌다..."

"""

    try:
        # 파일 업로드 처리
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

        # 메시지 처리
        messages = [{"role": "system", "content": system_prefix + system_message}]
        
        # 이전 대화 히스토리 추가
        if history:
            for user_msg, assistant_msg in history:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": assistant_msg})
        
        messages.append({"role": "user", "content": message})

        # API 호출 및 응답 처리
        client = get_client()
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
                value=initial_history,  # 저장된 히스토리로 초기화
                height=600, 
                label="대화창 💬",
                show_label=True
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
            ["흥미로운 소재 10가지를 제시해줘요 🤝"],
            ["더욱 자극적이고 묘사를 자세히해줘요 📚"],
            ["조선시대 배경으로 해줘요 🎯"],
            ["금기된 욕망을 알려줘요 ✨"],
            ["계속 이어서 작성해줘 🤔"],
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