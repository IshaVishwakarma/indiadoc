from langchain_community.document_loaders import PyPDFLoader

def load_pdf(path):
    loader = PyPDFLoader(path)
    docs = loader.load()

    print(f"Total pages: {len(docs)}")

    print("\n--- FIRST PAGE CONTENT ---\n")
    print(docs[0].page_content[:1000])

    print("\n--- METADATA ---\n")
    print(docs[0].metadata)


if __name__ == "__main__":
    load_pdf("data/dpdp.pdf")