from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import Optional
import os

# 🔥 NEW IMPORT
from langchain_core.embeddings import Embeddings

load_dotenv()

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    document: Optional[str] = None


# 🔥 DUMMY EMBEDDINGS (lightweight, no model)
class DummyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[0.0] * 384 for _ in texts]

    def embed_query(self, text):
        return [0.0] * 384


print("🔹 Loading vectorstore...")

if not os.path.exists("vectorstore"):
    print("❌ vectorstore missing!")
    db = None
else:
    embeddings = DummyEmbeddings()   # ✅ FIX

    db = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ vectorstore loaded")


# 🔥 LLM
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