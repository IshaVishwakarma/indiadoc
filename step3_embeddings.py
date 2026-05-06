from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_embeddings(pdf_path):
    # Load
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    print(f"Chunks ready: {len(chunks)}")

    # Embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Convert to vector store
    db = FAISS.from_documents(chunks, embeddings)

    # Save locally
    db.save_local("vectorstore")

    print("✅ Embeddings created and stored!")


if __name__ == "__main__":
    create_embeddings("data/dpdp.pdf")