from fastapi import FastAPI
from pydantic import BaseModel
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


# 🔥 LOAD VECTORSTORE ONLY (NO EMBEDDINGS)
print("🔹 Loading vectorstore...")

if not os.path.exists("vectorstore"):
    print("❌ vectorstore missing!")
    db = None
else:
    db = FAISS.load_local(
        "vectorstore",
        embeddings=None,   # ✅ KEY FIX (no embedding model)
        allow_dangerous_deserialization=True
    )
    print("✅ vectorstore loaded")


# 🔥 LOAD LLM (lightweight)
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0
)


@app.get("/")
def root():
    return {"status": "API running"}


@app.post("/query")
def query_docs(request: QueryRequest):
    try:
        if db is None:
            return {"answer": "Vectorstore not loaded", "sources": []}

        query = request.question.lower()

        # 🔥 retrieval (works without embeddings model)
        docs = db.similarity_search(query, k=4)

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
Answer ONLY from the context.

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
        return {
            "answer": f"Error: {str(e)}",
            "sources": []
        }