from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

def ask_question(question):
    # Load embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load FAISS index
    db = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Retrieve top chunks
    docs = db.docs = db.max_marginal_relevance_search(question, k=8)

    # Build context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Prompt
    prompt = f"""
You are a legal assistant.

STRICT RULES:
- Use ONLY the provided context
- Do NOT add any external knowledge
- Do NOT infer or assume anything
- If information is incomplete, say "Not specified in the document"

TASK:
Extract the exact answer from the context.

Format:
- Quote relevant lines
- Mention section numbers if present


Context:
{context}

Question:
{question}
"""
    print("\n🔍 RETRIEVED CONTEXT:\n")
    for i, doc in enumerate(docs):
     print(f"\n--- DOC {i+1} (Page {doc.metadata.get('page')}) ---")
     print(doc.page_content[:300])
    # Groq LLM
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    response = llm.invoke(prompt)

    print("\n🧠 ANSWER:\n")
    print(response.content)

    # Print sources (preview)
    print("\n📚 SOURCES:\n")

    for i, doc in enumerate(docs):
     print(f"\n--- Source {i+1} ---")
     print(f"Page: {doc.metadata.get('page')}")
     print(doc.page_content[:200])

if __name__ == "__main__":
    ask_question("What does the DPDP Act say about penalties, fines, or punishment for violations?")