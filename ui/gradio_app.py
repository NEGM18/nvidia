"""
Gradio UI — Premium interface for the Document Q&A Assistant.

Three tabs:
  1. Upload — Drag & drop PDF/DOCX with processing feedback
  2. Chat — Conversational Q&A with source citations
  3. Summary — One-click document summarization
"""

import os
import sys
import uuid
import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import UPLOAD_PATH

from app.ingestion.parser import parse_file
from app.ingestion.chunker import chunk_documents, get_chunk_stats
from app.vectorstore.store import VectorStoreManager
from app.chains.rag_chain import build_rag_chain
from app.chains.conversation import get_session_history, clear_session
from app.chains.summarizer import summarize_documents
from app.guardrails.safety import GuardRails

# ──────────────────────────────────────────────
# Shared State
# ──────────────────────────────────────────────
store_manager = VectorStoreManager()
guardrails = GuardRails()
current_documents = []
current_file_name = ""
rag_chain = None
session_id = str(uuid.uuid4())[:8]


# ──────────────────────────────────────────────
# Backend Functions
# ──────────────────────────────────────────────

def process_upload(file):
    """Process an uploaded file through the ingestion pipeline."""
    global current_documents, current_file_name, rag_chain

    if file is None:
        return "⚠️ No file selected. Please upload a PDF or DOCX file.", ""

    file_path = file.name if hasattr(file, 'name') else str(file)

    try:
        # Step 1: Parse
        documents = parse_file(file_path)
        if not documents:
            return "❌ No text could be extracted from this file.", ""

        # Step 2: Chunk
        chunks = chunk_documents(documents)
        stats = get_chunk_stats(chunks)

        # Step 3: Create vector store
        store_manager.create_from_documents(chunks)

        # Step 4: Build RAG chain
        retriever = store_manager.get_retriever()
        rag_chain = build_rag_chain(retriever)

        # Update state
        current_documents = chunks
        current_file_name = os.path.basename(file_path)

        # Format success message
        status_msg = f"""✅ **Document Processed Successfully!**

📄 **File:** {current_file_name}
📊 **Statistics:**
- Total chunks: **{stats['total_chunks']}**
- Average chunk size: **{stats['avg_chunk_size']:.0f}** characters
- Total characters: **{stats['total_characters']:,}**
- Min chunk: **{stats['min_chunk_size']}** | Max chunk: **{stats['max_chunk_size']}** chars

🟢 **Ready for questions!** Switch to the **Chat** tab to start asking."""

        details = f"Processed {len(documents)} pages/sections into {stats['total_chunks']} searchable chunks."

        return status_msg, details

    except ValueError as e:
        return f"❌ **Error:** {str(e)}", ""
    except Exception as e:
        return f"❌ **Processing Failed:** {str(e)}", ""


def chat_respond(message, history):
    """Handle a chat message with RAG pipeline."""
    global rag_chain

    if not store_manager.is_ready or rag_chain is None:
        return guardrails.get_no_document_message()

    # Guard-rail: validate input
    is_safe, rejection_msg = guardrails.check_input(message)
    if not is_safe:
        return rejection_msg

    try:
        # Retrieve with scores for guard-rail check
        results_with_scores = store_manager.search_with_scores(message)
        filtered_docs, has_relevant = guardrails.check_retrieval(results_with_scores)

        if not has_relevant:
            return guardrails.get_no_results_message()

        # Get conversation history
        chat_history = get_session_history(session_id)

        # Run the RAG chain
        answer = rag_chain.invoke(
            {"question": message, "history": chat_history.messages}
        )

        # Update history
        chat_history.add_user_message(message)
        chat_history.add_ai_message(answer)

        # Format with sources
        source_info = []
        for doc in filtered_docs[:3]:  # Show top 3 sources
            src = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            section = doc.metadata.get("section", "")
            loc = f"Page {page}" if page else f"Section {section}" if section else ""
            source_info.append(f"📌 {src} — {loc}" if loc else f"📌 {src}")

        sources_text = "\n".join(source_info)
        formatted = f"{answer}\n\n---\n**Sources:**\n{sources_text}\n"

        return formatted

    except Exception as e:
        return f"❌ **Error:** {str(e)}\n\nPlease try rephrasing your question."


def generate_summary():
    """Generate a summary of the uploaded document."""
    if not current_documents:
        return "📄 No document uploaded. Please upload a document in the **Upload** tab first."

    try:
        summary = summarize_documents(current_documents)
        return f"""## 📋 Document Summary: {current_file_name}

{summary}
"""

    except Exception as e:
        return f"❌ **Summarization Failed:** {str(e)}"


def clear_everything():
    """Clear all data and reset the session."""
    global current_documents, current_file_name, rag_chain, session_id

    store_manager.clear()
    clear_session(session_id)
    current_documents = []
    current_file_name = ""
    rag_chain = None
    session_id = str(uuid.uuid4())[:8]

    return "🗑️ All data cleared. Ready for a new document.", "", None


def get_system_status():
    """Get current system status."""
    if store_manager.is_ready:
        return f"""🟢 **System Ready**
- Document: **{current_file_name}**
- Chunks indexed: **{store_manager.doc_count}**
- Session: `{session_id}`"""
    else:
        return "🔴 **No document loaded.** Upload a file to get started."


# ──────────────────────────────────────────────
# Premium CSS Theme
# ──────────────────────────────────────────────
CUSTOM_CSS = """
/* Global Styling */
.gradio-container {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
}

/* Header */
.app-header {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: white;
    padding: 2rem;
    border-radius: 16px;
    margin-bottom: 1rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.app-header h1 {
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(90deg, #a78bfa, #06b6d4, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.app-header p {
    color: #cbd5e1;
    margin-top: 0.5rem;
    font-size: 1rem;
}

/* Tab Styling */
.tab-nav button {
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 12px 24px !important;
    border-radius: 12px 12px 0 0 !important;
    transition: all 0.3s ease !important;
}
.tab-nav button.selected {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    box-shadow: 0 -4px 12px rgba(99, 102, 241, 0.3) !important;
}

/* Upload Area */
.upload-area {
    border: 2px dashed #6366f1 !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    transition: all 0.3s ease !important;
    background: rgba(99, 102, 241, 0.05) !important;
}
.upload-area:hover {
    border-color: #8b5cf6 !important;
    background: rgba(99, 102, 241, 0.1) !important;
    transform: translateY(-2px) !important;
}

/* Status Cards */
.status-card {
    background: linear-gradient(135deg, #1e1b4b, #312e81) !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    color: white !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2) !important;
}

/* Chat Messages */
.message {
    border-radius: 12px !important;
    padding: 1rem !important;
    margin: 0.5rem 0 !important;
    animation: fadeIn 0.3s ease !important;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Buttons */
.primary-btn {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
}
.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
}

.danger-btn {
    background: linear-gradient(135deg, #ef4444, #dc2626) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

/* Summary Output */
.summary-output {
    background: rgba(99, 102, 241, 0.05) !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
}

/* Footer */
.app-footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.85rem;
    padding: 1rem;
    margin-top: 1rem;
}
"""


# ──────────────────────────────────────────────
# Build Gradio Interface
# ──────────────────────────────────────────────
def create_ui():
    """Build the Gradio interface."""

    with gr.Blocks(
        title="Document Q&A Assistant",
    ) as demo:

        # ── Header ──
        gr.HTML("""
        <div class="app-header">
            <h1>📄 Document Q&A Assistant</h1>
            <p>Upload documents & files • Ask questions with AI • Get cited answers</p>
        </div>
        """)

        with gr.Tabs() as tabs:

            # ════════════════════════════════════════
            # TAB 1: Upload
            # ════════════════════════════════════════
            with gr.Tab("📤 Upload Document", id="upload"):
                gr.Markdown("""
                ### Upload Your Document
                Drag & drop or click to upload a **PDF**, **DOCX**, or **TXT** file.
                The document will be parsed, chunked, and indexed for intelligent Q&A.
                """)

                with gr.Row():
                    with gr.Column(scale=2):
                        file_input = gr.File(
                            label="Drop your document here",
                            file_types=[".pdf", ".docx", ".doc", ".txt"],
                            type="filepath",
                            elem_classes=["upload-area"],
                        )
                        upload_btn = gr.Button(
                            "🚀 Process Document",
                            variant="primary",
                            elem_classes=["primary-btn"],
                            size="lg",
                        )

                    with gr.Column(scale=3):
                        upload_status = gr.Markdown(
                            value="Waiting for document upload...",
                            label="Processing Status",
                        )
                        upload_details = gr.Textbox(
                            label="Details",
                            interactive=False,
                            lines=2,
                        )

                upload_btn.click(
                    fn=process_upload,
                    inputs=[file_input],
                    outputs=[upload_status, upload_details],
                )

            # ════════════════════════════════════════
            # TAB 2: Chat
            # ════════════════════════════════════════
            with gr.Tab("💬 Chat with Document", id="chat"):
                gr.Markdown("""
                ### Ask Questions About Your Document
                Get AI-powered answers with **source citations** from the uploaded document.
                The assistant will only answer based on the document content.
                """)

                chatbot_display = gr.Chatbot(
                    height=500,
                    placeholder="<strong>Upload a document first, then ask me anything about it!</strong>",
                )

                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="Ask about clauses, terms, obligations, parties...",
                        container=False,
                        scale=7,
                        show_label=False,
                    )
                    chat_submit_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    clear_chat_btn = gr.Button("🗑️ Clear Chat", size="sm")

                gr.Examples(
                    examples=[
                        "What is the main topic of this document?",
                        "Can you summarize the key points?",
                        "Who is the intended audience for this document?",
                        "Are there any specific dates or deadlines mentioned?",
                        "What are the main conclusions?",
                        "Are there any action items?"
                    ],
                    inputs=chat_input,
                )

                def user_chat(message, history):
                    """Add user message and get bot response."""
                    if not message.strip():
                        return "", history
                    history = history or []
                    history.append({"role": "user", "content": message})
                    bot_response = chat_respond(message, history)
                    history.append({"role": "assistant", "content": bot_response})
                    return "", history

                chat_submit_btn.click(
                    fn=user_chat,
                    inputs=[chat_input, chatbot_display],
                    outputs=[chat_input, chatbot_display],
                )
                chat_input.submit(
                    fn=user_chat,
                    inputs=[chat_input, chatbot_display],
                    outputs=[chat_input, chatbot_display],
                )
                clear_chat_btn.click(
                    fn=lambda: ([], ""),
                    outputs=[chatbot_display, chat_input],
                )

            # ════════════════════════════════════════
            # TAB 3: Summary
            # ════════════════════════════════════════
            with gr.Tab("📋 Document Summary", id="summary"):
                gr.Markdown("""
                ### Get a Comprehensive Summary
                Generate a structured summary of the uploaded document covering
                key parties, terms, obligations, and notable clauses.
                """)

                with gr.Row():
                    summarize_btn = gr.Button(
                        "📝 Generate Summary",
                        variant="primary",
                        elem_classes=["primary-btn"],
                        size="lg",
                    )

                summary_output = gr.Markdown(
                    value="Upload a document and click **Generate Summary** to get started.",
                    label="Document Summary",
                    elem_classes=["summary-output"],
                )

                summarize_btn.click(
                    fn=generate_summary,
                    inputs=[],
                    outputs=[summary_output],
                )

            # ════════════════════════════════════════
            # TAB 4: Settings & Status
            # ════════════════════════════════════════
            with gr.Tab("⚙️ Settings", id="settings"):
                gr.Markdown("### System Status & Controls")

                with gr.Row():
                    with gr.Column():
                        status_display = gr.Markdown(
                            value=get_system_status(),
                            label="Status",
                        )
                        refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
                        refresh_btn.click(
                            fn=get_system_status,
                            inputs=[],
                            outputs=[status_display],
                        )

                    with gr.Column():
                        gr.Markdown("### ⚠️ Danger Zone")
                        clear_btn = gr.Button(
                            "🗑️ Clear All Data & Reset",
                            variant="stop",
                            elem_classes=["danger-btn"],
                        )
                        clear_status = gr.Markdown("")
                        clear_btn.click(
                            fn=clear_everything,
                            inputs=[],
                            outputs=[clear_status, upload_details, file_input],
                        )

        # ── Footer ──
        gr.HTML("""
        <div class="app-footer">
            <p>Document Q&A Assistant v1.0 • Powered by LangChain, FAISS & Gemini</p>
        </div>
        """)

    return demo


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
    )
