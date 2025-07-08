# app.py (Upgraded UI Version)

import gradio as gr
import time
from src.rag_pipeline import RAGPipeline # Assuming API version for speed

# --- 1. Initialize the RAG Pipeline (Done once) ---
print("Initializing the complaint analysis engine...")
rag_pipeline = RAGPipeline()
print("Engine ready.")

# --- 2. Define Core Logic Functions (Unchanged) ---
def ask_question(question, history):
    history = history or []
    # No need to yield user question, Gradio handles this with avatars
    
    # Query the RAG pipeline
    result = rag_pipeline.query(question)
    answer = result['answer']
    
    # Stream the answer
    response = ""
    for char in answer:
        response += char
        time.sleep(0.01)
        yield response, gr.Accordion(visible=False) # Keep sources hidden during stream
    
    yield response, gr.Accordion("View Sources", visible=True, open=False)

def show_sources(question):
    """Fetches and formats sources based on the last question."""
    if not question:
        return gr.Markdown(visible=False)
        
    result = rag_pipeline.query(question)
    sources = result['sources']
    
    formatted_sources = "### Sources Used to Generate Answer:\n\n"
    for i, src in enumerate(sources):
        src_text = src.replace('\n', ' ').replace('  ', ' ')
        formatted_sources += f"**Source {i+1}:**\n> {src_text}\n\n"
        
    return gr.Markdown(formatted_sources, visible=True)

# --- 3. Define the New Gradio UI ---

# Simple SVG for a logo
credistrust_logo = """
<div style="text-align: center; font-weight: bold; font-size: 24px; color: #0047AB;">
  <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-shield"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
  CrediTrust Complaint Analyzer
</div>
"""
# Avatars for the chatbot
bot_avatar = "https://raw.githubusercontent.com/gradio-app/gradio/main/guides/assets/logo.png"
user_avatar = "https://i.imgur.com/3o5hI6u.png" # Simple user icon

# Custom CSS for a professional look
custom_css = """
/* Main container */
.gradio-container { font-family: 'Inter', sans-serif; }
/* Chat message bubbles */
.message-bubble { background-color: #F0F4F8 !important; }
.message-bubble.user { background-color: #0047AB !important; color: white !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky"), css=custom_css, title="CrediTrust Analyzer") as app:
    
    # -- Header --
    gr.HTML(credistrust_logo)

    # -- Main Chat Interface --
    chatbot = gr.Chatbot(label="Chat History", height=500, type="messages")

    
    # To store the last question for fetching sources
    last_question = gr.State("")

    # -- Input Controls --
    with gr.Row():
        question_box = gr.Textbox(
            label="Your Question",
            placeholder="e.g., What are the main problems with money transfers?",
            scale=8,
            autofocus=True
        )
        submit_btn = gr.Button("➤", variant="primary", scale=1, min_width=50)

    # -- Sources Accordion --
    with gr.Accordion("View Sources", visible=False, open=False) as sources_accordion:
        sources_display = gr.Markdown()
    
    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
    
    # --- 4. Wire up the components ---
    
    def on_submit(question, history):
        # Store the question to state
        history.append([question, None])
        return question, history, "", gr.update(visible=False)

    def on_stream_and_show_sources(response):
        # This function gets called after the stream is complete
        # We use the stored question to fetch the sources
        return gr.update(visible=True)

    # Chain of events when a question is submitted
    question_box.submit(on_submit, [question_box, chatbot], [last_question, chatbot, question_box, sources_accordion]) \
        .then(ask_question, [last_question, chatbot], [chatbot.message, sources_accordion]) \
        .then(show_sources, [last_question], [sources_display])

    submit_btn.click(on_submit, [question_box, chatbot], [last_question, chatbot, question_box, sources_accordion]) \
        .then(ask_question, [last_question, chatbot], [chatbot.message, sources_accordion]) \
        .then(show_sources, [last_question], [sources_display])

    # Clear button functionality
    clear_btn.click(lambda: (None, None, "", None), outputs=[chatbot, last_question, question_box, sources_accordion])


# --- 5. Launch the App ---
if __name__ == "__main__":
    app.launch(debug=True)