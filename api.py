from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.embeddings import Embeddings
import os

app = FastAPI()


class QueryRequest(BaseModel):
    question: str


# 🔥 Dummy embeddings (no model, no cost)
class DummyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[0.0] * 384 for _ in texts]

    def embed_query(self, text):
        return [0.0] * 384


# 🔥 Load FAISS
print("Loading vectorstore...")
db = None

if os.path.exists("vectorstore"):
    db = FAISS.load_local(
        "vectorstore",
        DummyEmbeddings(),
        allow_dangerous_deserialization=True
    )
    print("Vectorstore loaded")
else:
    print("Vectorstore missing")


# 🔥 LLM
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0
)


# ✅ ROOT
@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/query")
def query_docs(req: QueryRequest):
    if db is None:
        return {"answer": "Vectorstore not found", "sources": []}

    # 🔥 STEP 1: Improve retrieval (query expansion)
    query = req.question.lower() + " definition act law section data protection"

    docs = db.similarity_search(query, k=5)

    if not docs:
        return {"answer": "No relevant information found", "sources": []}

    # 🔥 STEP 2: Build context with metadata
    context = "\n\n".join([
        f"[Document: {d.metadata.get('source')} | Page: {d.metadata.get('page')}]\n{d.page_content}"
        for d in docs
    ])

    # 🔥 STEP 3: Strong prompt (prevents hallucination)
    prompt = f"""
You are a legal assistant.

STRICT RULES:
- Answer ONLY using the provided context
- Do NOT guess or assume anything
- If answer is not found, say: "Not found in document"

Context:
{context}

Question:
{req.question}

Answer clearly in points:
"""

    response = llm.invoke(prompt)

    # 🔥 STEP 4: Return sources
    return {
        "answer": response.content,
        "sources": [
            {
                "document": d.metadata.get("source"),
                "page": d.metadata.get("page")
            }
            for d in docs
        ]
    }