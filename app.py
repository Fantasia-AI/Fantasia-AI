import gradio as gr
from huggingface_hub import InferenceClient
import os
import pandas as pd
from typing import List, Tuple

# LLM Models Definition
LLM_MODELS = {
    "Cohere c4ai-crp-08-2024": "CohereForAI/c4ai-command-r-plus-08-2024",  # Default
    "Meta Llama3.3-70B": "meta-llama/Llama-3.3-70B-Instruct"    # Backup model
}

def get_client(model_name="Cohere c4ai-crp-08-2024"):
    try:
        return InferenceClient(LLM_MODELS[model_name], token=os.getenv("HF_TOKEN"))
    except Exception:
        # If primary model fails, try backup model
        return InferenceClient(LLM_MODELS["Meta Llama3.3-70B"], token=os.getenv("HF_TOKEN"))

def analyze_file_content(content, file_type):
    """Analyze file content and return structural summary"""
    if file_type in ['parquet', 'csv']:
        try:
            lines = content.split('\n')
            header = lines[0]
            columns = header.count('|') - 1
            rows = len(lines) - 3
            return f"📊 Dataset Structure: {columns} columns, {rows} data samples"
        except:
            return "❌ Dataset structure analysis failed"
    
    lines = content.split('\n')
    total_lines = len(lines)
    non_empty_lines = len([line for line in lines if line.strip()])
    
    if any(keyword in content.lower() for keyword in ['def ', 'class ', 'import ', 'function']):
        functions = len([line for line in lines if 'def ' in line])
        classes = len([line for line in lines if 'class ' in line])
        imports = len([line for line in lines if 'import ' in line or 'from ' in line])
        return f"💻 Code Structure: {total_lines} lines (Functions: {functions}, Classes: {classes}, Imports: {imports})"
    
    paragraphs = content.count('\n\n') + 1
    words = len(content.split())
    return f"📝 Document Structure: {total_lines} lines, {paragraphs} paragraphs, ~{words} words"

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
                    content = f"📊 Data Preview:\n{df.head(10).to_markdown(index=False)}\n\n"
                    content += f"\n📈 Data Information:\n"
                    content += f"- Total Rows: {len(df)}\n"
                    content += f"- Total Columns: {len(df.columns)}\n"
                    content += f"- Column List: {', '.join(df.columns)}\n"
                    content += f"\n📋 Column Data Types:\n"
                    for col, dtype in df.dtypes.items():
                        content += f"- {col}: {dtype}\n"
                    null_counts = df.isnull().sum()
                    if null_counts.any():
                        content += f"\n⚠️ Missing Values:\n"
                        for col, null_count in null_counts[null_counts > 0].items():
                            content += f"- {col}: {null_count} missing\n"
                    return content, "csv"
                except UnicodeDecodeError:
                    continue
            raise UnicodeDecodeError(f"❌ Unable to read file with supported encodings ({', '.join(encodings)})")
        else:
            encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1']
            for encoding in encodings:
                try:
                    with open(file.name, 'r', encoding=encoding) as f:
                        content = f.read()
                    return content, "text"
                except UnicodeDecodeError:
                    continue
            raise UnicodeDecodeError(f"❌ Unable to read file with supported encodings ({', '.join(encodings)})")
    except Exception as e:
        return f"❌ Error reading file: {str(e)}", "error"

def format_history(history):
    formatted_history = []
    for user_msg, assistant_msg in history:
        formatted_history.append({"role": "user", "content": user_msg})
        if assistant_msg:
            formatted_history.append({"role": "assistant", "content": assistant_msg})
    return formatted_history

# 시스템 프롬프트 수정
def chat(message, history, uploaded_file, system_message="", max_tokens=4000, temperature=0.7, top_p=0.9):
    system_prefix = """저는 여러분의 친근하고 지적인 AI 어시스턴트입니다. 다음과 같은 원칙으로 소통하겠습니다:

1. 🤝 친근하고 공감적인 태도로 대화
2. 💡 명확하고 이해하기 쉬운 설명 제공
3. 🎯 질문의 의도를 정확히 파악하여 맞춤형 답변
4. 📚 필요한 경우 업로드된 파일 내용을 참고하여 구체적인 도움 제공
5. ✨ 추가적인 통찰과 제안을 통한 가치 있는 대화

항상 예의 바르고 친절하게 응답하며, 필요한 경우 구체적인 예시나 설명을 추가하여 
이해를 돕겠습니다."""

    if uploaded_file:
        content, file_type = read_uploaded_file(uploaded_file)
        if file_type == "error":
            return "", [{"role": "user", "content": message}, {"role": "assistant", "content": content}]
        
        file_summary = analyze_file_content(content, file_type)
        
        if file_type in ['parquet', 'csv']:
            system_message += f"\n\n파일 내용:\n```markdown\n{content}\n```"
        else:
            system_message += f"\n\n파일 내용:\n```\n{content}\n```"
            
        if message == "Starting file analysis...":
            message = f"""[파일 구조 분석] {file_summary}

다음 관점에서 도움을 드리겠습니다:
1. 📋 전반적인 내용 파악
2. 💡 주요 특징 설명
3. 🎯 실용적인 활용 방안
4. ✨ 개선 제안
5. 💬 추가 질문이나 필요한 설명"""

    messages = [{"role": "system", "content": f"{system_prefix} {system_message}"}]
    
    if history is not None:
        for item in history:
            if isinstance(item, dict):
                messages.append(item)
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                messages.append({"role": "user", "content": item[0]})
                if item[1]:
                    messages.append({"role": "assistant", "content": item[1]})

    messages.append({"role": "user", "content": message})

    try:
        client = get_client()
        partial_message = ""
        current_history = []
        
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
                current_history = [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": partial_message}
                ]
                yield "", current_history
                
    except Exception as e:
        error_msg = f"❌ 오류가 발생했습니다: {str(e)}"
        error_history = [
            {"role": "user", "content": message},
            {"role": "assistant", "content": error_msg}
        ]
        yield "", error_history

# UI 텍스트 한글화
with gr.Blocks(theme="Yntec/HaleyCH_Theme_Orange", title="GiniGEN 🤖") as demo:
    gr.HTML(
        """
        <div style="text-align: center; max-width: 800px; margin: 0 auto;">
            <h1 style="font-size: 3em; font-weight: 600; margin: 0.5em;">AI 어시스턴트 🤖</h1>
            <h3 style="font-size: 1.2em; margin: 1em;">당신의 든든한 대화 파트너 💬</h3>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                height=600, 
                label="대화창 💬",
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
            gr.Markdown("### 파일 업로드 📁\n지원 형식: 텍스트, 코드, CSV, Parquet 파일")
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

    # 예시 질문 수정
    gr.Examples(
        examples=[
            ["안녕하세요! 어떤 도움이 필요하신가요? 🤝"],
            ["이 내용에 대해 좀 더 자세히 설명해 주실 수 있나요? 💡"],
            ["제가 이해하기 쉽게 설명해 주시겠어요? 📚"],
            ["이 내용을 실제로 어떻게 활용할 수 있을까요? 🎯"],
            ["추가로 조언해 주실 내용이 있으신가요? ✨"],
            ["궁금한 점이 더 있는데 여쭤봐도 될까요? 🤔"],
        ],
        inputs=msg,
    )

if __name__ == "__main__":
    demo.launch()