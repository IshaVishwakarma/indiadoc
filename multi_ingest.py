import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_FOLDER = "data"
VECTOR_PATH = "vectorstore"

def ingest_all():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    all_chunks = []

    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".pdf"):
            path = os.path.join(DATA_FOLDER, file)

            loader = PyPDFLoader(path)
            docs = loader.load()

            chunks = splitter.split_documents(docs)

            # ✅ IMPORTANT: add metadata
            for chunk in chunks:
                chunk.metadata["source"] = file

            all_chunks.extend(chunks)

            print(f"✅ Processed: {file}")

    # ✅ Create fresh DB
    db = FAISS.from_documents(all_chunks, embeddings)
    db.save_local(VECTOR_PATH)

    print("🚀 All documents indexed successfully!")


if __name__ == "__main__":
    ingest_all()