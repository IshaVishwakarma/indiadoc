from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def query_docs(question):
    # Load embedding model (same as before!)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load FAISS index
    db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

    # Search similar chunks
    docs = db.similarity_search(question, k=3)

    print("\n🔍 Top Results:\n")

    for i, doc in enumerate(docs):
        print(f"\n--- RESULT {i+1} ---")
        print("Page:", doc.metadata.get("page"))
        print(doc.page_content[:300])


if __name__ == "__main__":
    query_docs("What is the penalty under the DPDP Act?")