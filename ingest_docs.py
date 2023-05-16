from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

import os
import dotenv

dotenv.load_dotenv()

pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"),
              environment=os.environ.get("PINECONE_ENVIRONMENT_REGION"))


def ingest_docs():
    loader = ReadTheDocsLoader(path="langchain-docs/python.langchain.com/en/latest")
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])

    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Split document into {len(documents)} chunks")

    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("langchain-docs/", "https://")
        doc.metadata.update({"source": new_url})

    print(f"Going to insert {len(documents)} chunks into Pinecone.")

    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(documents=documents, embedding=embeddings, index_name="langchain-docs-db")


if __name__ == "__main__":
    ingest_docs()
