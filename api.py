from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import Optional
import os

from langchain_core.embeddings import Embeddings

load_dotenv()

app = FastAPI()


class QueryRequest(BaseModel):
    question: str
    document: Optional[str] = None


# 🔥 Dummy embeddings (no model, no memory)
class DummyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[0.0] * 384 for _ in texts]

    def embed_query(self, text):
        return [0.0] * 384


# 🔥 Load vectorstore ONCE
print("🔹 Loading vectorstore...")

db = None
if os.path.exists("vectorstore"):
    embeddings = DummyEmbeddings()
    db = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ vectorstore loaded")
else:
    print("❌ vectorstore missing!")


# 🔥 LLM
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0
)


# ✅ ROOT (Railway health check friendly)
@app.get("/")
def root():
    return {"status": "ok", "message": "API is running"}


# ✅ EXTRA HEALTH ENDPOINT (important)
@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/query")
def query_docs(request: QueryRequest):
    try:
        if db is None:
            return {"answer": "Vectorstore not loaded", "sources": []}

        query = request.question.strip().lower()

        # 🔥 Retrieval (fast)
        docs = db.similarity_search(query, k=4)

        if not docs:
            return {"answer": "No relevant information found.", "sources": []}

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
Answer ONLY from the context.

Context:
{context}

Question:
{request.question}
"""

        # 🔥 LLM call
        response = llm.invoke(prompt)

        return {
            "answer": response.content,
            "sources": [
                {
                    "document": doc.metadata.get("source"),
                    "page": doc.metadata.get("page")
                }
                for doc in docs
            ]
        }

    except Exception as e:
        print("❌ ERROR:", str(e))
        return {
            "answer": "Something went wrong. Please try again.",
            "sources": []
        }