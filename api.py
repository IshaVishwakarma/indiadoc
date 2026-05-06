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

# 🔥 GLOBAL LOAD (ONLY ONCE)
print("🔹 Loading embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("🔹 Checking vectorstore...")
if not os.path.exists("vectorstore"):
    print("❌ vectorstore missing!")
    db = None
else:
    print("🔹 Loading vectorstore...")
    db = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ Vectorstore loaded")

# 🔥 LLM INIT ONCE
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0
)


@app.post("/query")
def query_docs(request: QueryRequest):
    try:
        if db is None:
            return {"answer": "Vectorstore not loaded", "sources": []}

        query = request.question.lower()

        # 🔥 Auto detect doc
        if not request.document:
            if "it act" in query:
                request.document = "it_act.pdf"
            elif "dpdp" in query:
                request.document = "dpdp.pdf"
            elif "rbi" in query:
                request.document = "rbi_guidelines.pdf"

        # 🔥 Detect penalty query
        is_penalty_query = any(word in query for word in [
            "penalty", "fine", "punishment"
        ])

        if is_penalty_query:
            query += " penalty fine imprisonment section 33"

        # 🔥 Retrieval
        if request.document:
            docs = db.similarity_search(
                query,
                k=6,
                filter={"source": request.document}
            )
        else:
            docs = db.max_marginal_relevance_search(
                query,
                k=6,
                fetch_k=20
            )

        # 🔥 Build context
        context = "\n\n".join([
            doc.page_content for doc in docs
        ])

        prompt = f"""
Answer the question based only on context.

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