import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
from app.llm import generate_llm_answer  # Corrected import

# Load sentence embedding model once (MiniLM is efficient and fast)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def load_document(path):
    """
    Load a PDF and extract text page by page.
    """
    doc = fitz.open(path)
    return [page.get_text() for page in doc]

def embed_pages(pages):
    """
    Compute embeddings for all pages of the document.
    """
    return embedding_model.encode(pages, convert_to_tensor=True)

def get_top_k_pages(question, pages, page_embeddings, k=3):
    """
    Get top-k most relevant pages based on cosine similarity to the question.
    """
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    similarities = util.cos_sim(question_embedding, page_embeddings)[0]
    top_indices = similarities.argsort(descending=True)[:k]
    return [(i.item(), pages[i]) for i in top_indices]

def answer_question_with_rag(question, pages, page_embeddings):
    """
    Complete RAG pipeline: retrieve top pages, create context, generate answer using LLM.
    """
    top_pages = get_top_k_pages(question, pages, page_embeddings)

    context = "\n\n".join([text for _, text in top_pages])
    page_numbers = [i + 1 for i, _ in top_pages]
    snippets = [text.strip()[:500] + "..." for _, text in top_pages]

    raw_answer = generate_llm_answer(context, question)

    # Rule-based logic (can be upgraded with classification model later)
    decision = "approved" if "covered" in raw_answer.lower() or "reimbursed" in raw_answer.lower() else "rejected"
    amount = "Depends on policy terms"  # You can add NER/regex to extract actual amounts

    return {
        "decision": decision,
        "amount": amount,
        "justification": raw_answer,
        "source_pages": page_numbers,
        "source_snippets": snippets
    }
