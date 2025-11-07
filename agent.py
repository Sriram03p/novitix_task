import streamlit as st
import google.generativeai as genai
import faiss
import numpy as np
from PyPDF2 import PdfReader

# ------------------------------
# CONFIGURE GEMINI API
# ------------------------------
GEMINI_API_KEY = "AIzaSyCQhvcaawVDZG6E1t7v_57oZMr2RTY0OPg"
genai.configure(api_key=GEMINI_API_KEY)

EMBED_MODEL = "text-embedding-004"
LLM_MODEL = "gemini-flash-latest"


# ------------------------------
# FUNCTIONS
# ------------------------------
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


def embed_text(texts):
    embeddings = []
    for txt in texts:
        result = genai.embed_content(model=EMBED_MODEL, content=txt)
        embeddings.append(result["embedding"])
    return np.array(embeddings, dtype="float32")


def search_index(query, index, chunks, k=3):
    q_emb = genai.embed_content(model=EMBED_MODEL, content=query)["embedding"]
    q_emb = np.array([q_emb], dtype="float32")
    D, I = index.search(q_emb, k)
    return [chunks[i] for i in I[0]]


def ask_gemini(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""
    Use the policy PDF context below to answer the question.

    CONTEXT:
    {context}

    QUESTION:
    {query}

    If answer is not found in the context, say:
    "I could not find the answer in the PDF."
    """
    model = genai.GenerativeModel(LLM_MODEL)
    response = model.generate_content(prompt)
    return response.text


def governance_agent(action, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
    You are a Governance Compliance Agent.

    Evaluate the proposed action using ONLY the provided policy context.

    CONTEXT:
    {context}

    ACTION:
    {action}

    ✅ IMPORTANT INSTRUCTIONS:
    - Return ONLY valid JSON.
    - Do NOT add backticks.
    - Do NOT add code blocks.
    - Do NOT add explanations.
    - Output MUST be directly parseable JSON.

    REQUIRED JSON FORMAT (no markdown):
    {{
        "decision": "",
        "reason": "",
        "suggested_changes": [],
        "references": []
    }}
    """

    model = genai.GenerativeModel(LLM_MODEL)
    response = model.generate_content(prompt)
    return response.text


# ------------------------------
# STREAMLIT APP UI
# ------------------------------
st.title("📘 Policy RAG System + 🛡 Governance Agent")

# ✅ Upload PDF (Same for both sections)
uploaded_file = st.file_uploader("Upload policy/regulatory PDF", type=["pdf"])

# Tabs for separating features
tab1, tab2 = st.tabs(["📄 RAG Q&A", "🛡 Governance Agent"])

if uploaded_file:
    raw_text = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(raw_text)
    embeddings = embed_text(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    st.success("✅ PDF processed successfully!")

    # ---------------------------------------------------------
    # ✅ TAB 1 — RAG QUESTION ANSWERING
    # ---------------------------------------------------------
    with tab1:
        st.header("📄 RAG — Ask Questions About the PDF")

        query = st.text_input("Ask your question:")

        if query:
            retrieved_chunks = search_index(query, index, chunks, k=3)
            answer = ask_gemini(query, retrieved_chunks)

            st.subheader("💡 Answer")
            st.write(answer)

            st.subheader("📑 Retrieved Context")
            with st.expander("Show chunks"):
                for c in retrieved_chunks:
                    st.write(c)

    # ---------------------------------------------------------
    # ✅ TAB 2 — GOVERNANCE AGENT
    # ---------------------------------------------------------
    with tab2:
        st.header("🛡 Governance Compliance Agent")

        action = st.text_input("Describe the action to evaluate:")

        if action:
            if "retrieved_chunks" in locals():
                governance_output = governance_agent(action, retrieved_chunks)
                st.subheader("✅ Governance Decision")
                st.json(governance_output)
            else:
                st.warning("❗ Please ask at least one question in the RAG tab to load context.")
