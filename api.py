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
print("🔹 Loading vectorstore...")
db = None

if os.path.exists("vectorstore"):
    db = FAISS.load_local(
        "vectorstore",
        DummyEmbeddings(),
        allow_dangerous_deserialization=True
    )
    print("✅ Vectorstore loaded")
else:
    print("❌ Vectorstore missing")


# 🔥 LLM
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0
)


# ✅ Health endpoints
@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/query")
def query_docs(req: QueryRequest):
    try:
        if db is None:
            return {"answer": "Vectorstore not found", "sources": []}

        # 🔥 STEP 1 — Strong query expansion
        query = req.question.lower() + """
        digital personal data protection act 2023 definition meaning objective introduction law india dpdp
        penalties punishment fine section
        data security privacy rules obligations
        """

        # 🔥 STEP 2 — Better retrieval (MMR)
        docs = db.max_marginal_relevance_search(
            query,
            k=8,
            fetch_k=25
        )

        if not docs:
            return {"answer": "No relevant information found", "sources": []}

        # 🔥 STEP 3 — Prioritize early pages (important for definitions)
        docs = sorted(docs, key=lambda x: x.metadata.get("page", 0))

        # 🔥 STEP 4 — Build structured context
        context = "\n\n".join([
            f"[Document: {d.metadata.get('source')} | Page: {d.metadata.get('page')}]\n{d.page_content}"
            for d in docs
        ])

        # 🔥 STEP 5 — Strong anti-hallucination prompt
        prompt = f"""
You are a legal assistant.

STRICT RULES:
- Answer ONLY using the provided context
- Do NOT guess or assume anything
- If exact answer is not present, say: "Not found in document"

INSTRUCTIONS:
- Prefer definitions and introductory sections
- Be precise and factual
- Use bullet points if needed

Context:
{context}

Question:
{req.question}

Answer:
"""

        response = llm.invoke(prompt)

        # 🔥 STEP 6 — Clean output
        return {
            "answer": response.content,
            "sources": [
                {
                    "document": d.metadata.get("source"),
                    "page": d.metadata.get("page")
                }
                for d in docs[:5]
            ]
        }

    except Exception as e:
        print("❌ ERROR:", str(e))
        return {
            "answer": "Something went wrong. Please try again.",
            "sources": []
        }