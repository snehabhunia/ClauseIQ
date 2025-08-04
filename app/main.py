from fastapi import FastAPI, Query
from app.rag_engine import load_document, embed_pages, answer_question_with_rag

app = FastAPI(
    title="RAG-based Question Answering System",
    description="Ask questions based on uploaded documents using LLM and vector embeddings",
    version="1.0"
)

# Load and embed documents once at startup
pdf_path = "data/BAJHLIP23020V012223 (1).pdf"
pages = load_document(pdf_path)
page_embeddings = embed_pages(pages)

@app.get("/ask")
def ask(question: str = Query(..., description="Your question")):
    answer = answer_question_with_rag(question, pages, page_embeddings)
    return {"question": question, "answer": answer}

