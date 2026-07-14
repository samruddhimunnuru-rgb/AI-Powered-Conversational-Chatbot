import gradio as gr

from rag import (
    extract_text,
    split_text,
    create_embeddings,
    create_faiss_index,
    search_similar_chunks
)

from chatbot import generate_response

# Global variables
chunks = None
faiss_index = None


def process_pdf(pdf):

    global chunks
    global faiss_index

    text = extract_text(pdf.name)

    chunks = split_text(text)

    embeddings = create_embeddings(chunks)

    faiss_index = create_faiss_index(embeddings)

    return "✅ PDF processed successfully! You can now ask questions."
def ask_question(question):

    global chunks
    global faiss_index

    if chunks is None or faiss_index is None:
        return "Please upload a PDF first."

    retrieved_chunks = search_similar_chunks(
        question,
        chunks,
        faiss_index
    )

    context = "\n\n".join(retrieved_chunks)

    answer = generate_response(
        question,
        context
    )

    return answer
with gr.Blocks(title="AI PDF Document Assistant") as demo:

    gr.Markdown("# 📄 AI PDF Document Assistant")

    pdf = gr.File(
        label="Upload PDF",
        file_types=[".pdf"]
    )

    upload_btn = gr.Button("Process PDF")

    status = gr.Textbox(
        label="Status"
    )

    question = gr.Textbox(
        label="Ask a Question",
        placeholder="Ask anything about your PDF..."
    )

    answer = gr.Textbox(
        label="Answer",
        lines=10
    )

    ask_btn = gr.Button("Get Answer")

    upload_btn.click(
        process_pdf,
        inputs=pdf,
        outputs=status
    )

    ask_btn.click(
        ask_question,
        inputs=question,
        outputs=answer
    )

demo.launch()