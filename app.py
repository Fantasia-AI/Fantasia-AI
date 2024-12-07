import gradio as gr
from huggingface_hub import InferenceClient
import os
import pandas as pd
from typing import List, Tuple

# LLM Models Definition
LLM_MODELS = {
    "Cohere c4ai-crp-08-2024": "CohereForAI/c4ai-command-r-plus-08-2024",  # Default
    "Meta Llama3.3-70B": "meta-llama/Llama-3.3-70B-Instruct",    
    "Mistral Nemo 2407": "mistralai/Mistral-Nemo-Instruct-2407",
    "Alibaba Qwen QwQ-32B": "Qwen/QwQ-32B-Preview"
}

def get_client(model_name):
    return InferenceClient(LLM_MODELS[model_name], token=os.getenv("HF_TOKEN"))

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

def chat(message, history, uploaded_file, model_name, system_message="", max_tokens=4000, temperature=0.7, top_p=0.9):
    system_prefix = """You are a file analysis expert. Analyze the uploaded file in depth from the following perspectives:
1. 📋 Overall structure and composition
2. 📊 Key content and pattern analysis
3. 📈 Data characteristics and meaning
   - For datasets: Column meanings, data types, value distributions
   - For text/code: Structural features, main patterns
4. 💡 Potential applications
5. ✨ Data quality and areas for improvement

Provide detailed and structured analysis from an expert perspective, but explain in an easy-to-understand way. Format the analysis results in Markdown and include specific examples where possible."""

    if uploaded_file:
        content, file_type = read_uploaded_file(uploaded_file)
        if file_type == "error":
            return "", [{"role": "user", "content": message}, {"role": "assistant", "content": content}]
        
        file_summary = analyze_file_content(content, file_type)
        
        if file_type in ['parquet', 'csv']:
            system_message += f"\n\nFile Content:\n```markdown\n{content}\n```"
        else:
            system_message += f"\n\nFile Content:\n```\n{content}\n```"
            
        if message == "Starting file analysis...":
            message = f"""[Structure Analysis] {file_summary}

Please provide detailed analysis from these perspectives:
1. 📋 Overall file structure and format
2. 📊 Key content and component analysis
3. 📈 Data/content characteristics and patterns
4. ⭐ Quality and completeness evaluation
5. 💡 Suggested improvements
6. 🎯 Practical applications and recommendations"""

    messages = [{"role": "system", "content": f"{system_prefix} {system_message}"}]
    
    # Convert history to message format
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        if h[1]:
            messages.append({"role": "assistant", "content": h[1]})
    
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
                new_history = history + [[message, partial_message]]
                formatted_history = []
                for h in new_history:
                    formatted_history.append({"role": "user", "content": h[0]})
                    if h[1]:
                        formatted_history.append({"role": "assistant", "content": h[1]})
                yield "", formatted_history
                
    except Exception as e:
        error_msg = f"❌ Inference error: {str(e)}"
        formatted_history = history + [[message, error_msg]]
        yield "", [{"role": "user", "content": h[0], "role": "assistant", "content": h[1]} for h in formatted_history]

css = """
footer {visibility: hidden}
"""

# ... (이전 코드 동일)

with gr.Blocks(theme="Yntec/HaleyCH_Theme_Orange", css=css, title="EveryChat 🤖") as demo:
    gr.HTML(
        """
        <div style="text-align: center; max-width: 800px; margin: 0 auto;">
            <h1 style="font-size: 3em; font-weight: 600; margin: 0.5em;">EveryChat 🤖</h1>
            <h3 style="font-size: 1.2em; margin: 1em;">Your Intelligent File Analysis Assistant 📊</h3>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                height=600, 
                label="Chat Interface 💬",
                type="messages"
            )
            msg = gr.Textbox(
                label="Type your message",
                show_label=False,
                placeholder="Ask me anything about the uploaded file... 💭",
                container=False
            )
            with gr.Row():
                clear = gr.ClearButton([msg, chatbot])
                send = gr.Button("Send 📤")
        
        with gr.Column(scale=1):
            model_name = gr.Radio(
                choices=list(LLM_MODELS.keys()),
                value="Cohere c4ai-crp-08-2024",
                label="Select LLM Model 🤖",
                info="Choose your preferred AI model"
            )
            
            gr.Markdown("### Upload File 📁\nSupport: Text, Code, CSV, Parquet files")
            file_upload = gr.File(
                label="Upload File",
                file_types=["text", ".csv", ".parquet"],
                type="filepath"
            )
            
            with gr.Accordion("Advanced Settings ⚙️", open=False):
                system_message = gr.Textbox(label="System Message 📝", value="")
                max_tokens = gr.Slider(minimum=1, maximum=8000, value=4000, label="Max Tokens 📊")
                temperature = gr.Slider(minimum=0, maximum=1, value=0.7, label="Temperature 🌡️")
                top_p = gr.Slider(minimum=0, maximum=1, value=0.9, label="Top P 📈")

    # Event bindings
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

    send.click(
        chat,
        inputs=[msg, chatbot, file_upload, model_name, system_message, max_tokens, temperature, top_p],
        outputs=[msg, chatbot],
        queue=True
    ).then(
        lambda: gr.update(interactive=True),
        None,
        [msg]
    )

    # Auto-analysis on file upload
    file_upload.change(
        chat,
        inputs=[gr.Textbox(value="Starting file analysis..."), chatbot, file_upload, model_name, system_message, max_tokens, temperature, top_p],
        outputs=[msg, chatbot],
        queue=True
    )

    # Example queries
    gr.Examples(
        examples=[
            ["Please explain the overall structure and features of the file in detail 📋"],
            ["Analyze the main patterns and characteristics of this file 📊"],
            ["Evaluate the file's quality and potential improvements 💡"],
            ["How can we practically utilize this file? 🎯"],
            ["Summarize the main content and derive key insights ✨"],
            ["Please continue with more detailed analysis 📈"],
        ],
        inputs=msg,
    )

if __name__ == "__main__":
    demo.launch()