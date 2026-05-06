from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    document: Optional[str] = None


@app.post("/query")
def query_docs(request: QueryRequest):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # 🔥 STEP 1: Normalize question
    query = request.question.lower()

    # 🔥 STEP 2: Auto-detect document (VERY IMPORTANT)
    if not request.document:
        if "it act" in query:
            request.document = "it_act.pdf"
        elif "dpdp" in query:
            request.document = "dpdp.pdf"
        elif "rbi" in query:
            request.document = "rbi_guidelines.pdf"

    # 🔥 STEP 3: Detect query type
    is_penalty_query = any(word in query for word in [
        "penalty", "penalties", "fine", "punishment"
    ])

    if is_penalty_query:
        query += " penalties punishment fine imprisonment section 33"

    # 🔥 STEP 4: Retrieval
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

    # 🔥 STEP 5: Remove duplicate chunks
    seen = set()
    unique_docs = []
    for doc in docs:
        key = (doc.page_content[:100], doc.metadata.get("source"))
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
    docs = unique_docs

    # 🔥 STEP 6: Smart filtering (only for penalty queries)
    if is_penalty_query:
        filtered_docs = [
            doc for doc in docs
            if any(keyword in doc.page_content.lower() for keyword in [
                "penalty", "penalties", "punishable", "imprisonment", "fine", "section 33"
            ])
        ]
        docs = filtered_docs if filtered_docs else docs

    # 🔥 STEP 7: Build context
    context = "\n\n".join([
        f"[Document: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}]\n{doc.page_content}"
        for doc in docs
    ])

    # 🔥 STEP 8: Adaptive instruction
    if is_penalty_query:
        instruction = """
- Extract ONLY penalty-related information
- Ignore unrelated sections
"""
    else:
        instruction = """
- Answer clearly in bullet points
- Focus only on relevant information
"""

    prompt = f"""
You are a legal assistant.

STRICT RULES:
- Use ONLY the provided context
- Do NOT add external knowledge

{instruction}

IMPORTANT:
- Group answers by document name (dpdp.pdf, it_act.pdf, rbi_guidelines.pdf)
- Do NOT confuse section names with document names

FORMAT:

[Document: <document_name>]
- Key points

If no relevant info is found:
"Not specified in the document"

Context:
{context}

Question:
{request.question}
"""

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    response = llm.invoke(prompt)

    # 🔥 STEP 9: Limit clean sources (UI polish)
    sources = [
        {
            "document": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "content": doc.page_content[:200]
        }
        for doc in docs[:4]   # show only top 4
    ]

    return {
        "answer": response.content,
        "sources": sources
    }