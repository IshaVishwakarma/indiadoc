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


@app.post("/query")
def query_docs(request: QueryRequest):
    try:
        print("🔹 Request received")

        # 🔥 STEP 1: Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # 🔥 STEP 2: Check vectorstore exists
        if not os.path.exists("vectorstore"):
            return {
                "answer": "Vectorstore not found on server",
                "sources": []
            }

        print("🔹 Loading vectorstore...")

        db = FAISS.load_local(
            "vectorstore",
            embeddings,
            allow_dangerous_deserialization=True
        )

        print("✅ Vectorstore loaded")

        # 🔥 STEP 3: Normalize query
        query = request.question.lower()

        # 🔥 STEP 4: Auto-detect document
        if not request.document:
            if "it act" in query:
                request.document = "it_act.pdf"
            elif "dpdp" in query:
                request.document = "dpdp.pdf"
            elif "rbi" in query:
                request.document = "rbi_guidelines.pdf"

        # 🔥 STEP 5: Detect penalty query
        is_penalty_query = any(word in query for word in [
            "penalty", "penalties", "fine", "punishment"
        ])

        if is_penalty_query:
            query += " penalties punishment fine imprisonment section 33"

        # 🔥 STEP 6: Retrieval
        if request.document:
            docs = db.similarity_search(
                query,
                k=8,
                filter={"source": request.document}
            )
        else:
            docs = db.max_marginal_relevance_search(
                query,
                k=10,
                fetch_k=40
            )

        # 🔥 STEP 7: Remove duplicates
        seen = set()
        unique_docs = []
        for doc in docs:
            key = (doc.page_content[:100], doc.metadata.get("source"))
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)
        docs = unique_docs

        # 🔥 STEP 8: Filter (only for penalty)
        if is_penalty_query:
            filtered_docs = [
                doc for doc in docs
                if any(keyword in doc.page_content.lower() for keyword in [
                    "penalty", "penalties", "punishable", "imprisonment", "fine", "section 33"
                ])
            ]
            docs = filtered_docs if filtered_docs else docs

        # 🔥 STEP 9: Build context
        context = "\n\n".join([
            f"[Document: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}]\n{doc.page_content}"
            for doc in docs
        ])

        # 🔥 STEP 10: Prompt
        instruction = (
            "Extract ONLY penalty-related information"
            if is_penalty_query
            else "Answer clearly in bullet points"
        )

        prompt = f"""
You are a legal assistant.

STRICT RULES:
- Use ONLY the provided context
- Do NOT add external knowledge

{instruction}

Context:
{context}

Question:
{request.question}
"""

        # 🔥 STEP 11: LLM
        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0
        )

        response = llm.invoke(prompt)

        # 🔥 STEP 12: Sources
        sources = [
            {
                "document": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "content": doc.page_content[:200]
            }
            for doc in docs[:4]
        ]

        return {
            "answer": response.content,
            "sources": sources
        }

    except Exception as e:
        print("❌ ERROR:", str(e))
        return {
            "answer": f"Server error: {str(e)}",
            "sources": []
        }