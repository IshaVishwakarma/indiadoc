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


# ✅ FAST ROOT (VERY IMPORTANT)
@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/query")
def query_docs(req: QueryRequest):
    if db is None:
        return {"answer": "Vectorstore not found"}

    docs = db.similarity_search(req.question, k=3)

    context = "\n\n".join([d.page_content for d in docs])

    response = llm.invoke(
        f"Answer from context:\n{context}\n\nQ: {req.question}"
    )

    return {"answer": response.content}