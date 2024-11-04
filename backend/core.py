import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama

load_dotenv()


def run_query(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OllamaEmbeddings(model="llama3.2")
    docsearch = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )
    chat = ChatOllama(model="llama3.2", verbose=True, temperature=0)

    retrieval_chat_promt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_chat_promt)

    rephrase_promt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_retrieval = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_promt
    )

    qa = create_retrieval_chain(
        retriever=history_retrieval, combine_docs_chain=stuff_documents_chain
    )

    response = qa.invoke(input={"input": query, "chat_history": chat_history})
    result = {
        "query": response["input"],
        "result": response["answer"],
        "source_documents": response["context"],
    }

    return result


if __name__ == "__main__":
    res = run_query("Como planea ayudar a los mas pobres?")
    print(res["result"])
