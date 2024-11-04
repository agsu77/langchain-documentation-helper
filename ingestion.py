from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()


def ingest_docs():
    loader: PyPDFLoader = PyPDFLoader(
        file_path="./files/FA-2025-2030.pdf", extract_images=True
    )
    raw_docs = loader.load()
    print(f"loaded {len(raw_docs)} documents")

    text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=50
    )

    chunks = text_splitter.split_documents(raw_docs)

    print(f"Going to add {len(chunks)} to Pinecone")
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="llama3.2"),
        index_name="langchain-doc-demo",
    )

    print("Loading to vectorestore DONE")


if __name__ == "__main__":
    ingest_docs()
