from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import Optional
import os

load_dotenv()

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    document: Optional[str] = None


# 🔥 LOAD ONLY WHEN NEEDED
def load_db():
    print("🔹 Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if not os.path.exists("vectorstore"):
        raise Exception("Vectorstore not found")

    print("🔹 Loading FAISS...")
    db = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db


@app.post("/query")
def query_docs(request: QueryRequest):
    try:
        db = load_db()   # 🔥 load inside request

        query = request.question.lower()

        docs = db.max_marginal_relevance_search(
            query,
            k=5,
            fetch_k=10
        )

        context = "\n\n".join([doc.page_content for doc in docs])

        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0
        )

        prompt = f"""
Answer based only on context.

Context:
{context}

Question:
{request.question}
"""

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
            "answer": f"Error: {str(e)}",
            "sources": []
        }