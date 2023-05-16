from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

import pinecone
import os

from constants import INDEX_NAME
from typing import Any, List, Tuple
import dotenv

dotenv.load_dotenv()

pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENVIRONMENT_REGION")
)


def run_llm(query: str, chat_history: List[Tuple[str, Any]] = None) -> Any:

    if chat_history is None:
        chat_history = []

    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)

    chat = ChatOpenAI(verbose=True, temperature=0)

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, chain_type="stuff", return_source_documents=True, retriever=docsearch.as_retriever()
    )

    return qa({"question": query, "chat_history": chat_history})

