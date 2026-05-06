from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def chunk_pdf(path):
    loader = PyPDFLoader(path)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages")

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    # Print sample chunks
    for i, chunk in enumerate(chunks[:3]):
        print("\n--- CHUNK", i, "---")
        print("Page:", chunk.metadata.get("page"))
        print(chunk.page_content[:300])

    return chunks


if __name__ == "__main__":
    chunk_pdf("data/dpdp.pdf")