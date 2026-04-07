from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_cohere import CohereRerank
from langchain_community.vectorstores.chroma import Chroma
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
import os
from llm import embeddings  # custom embeddings model from your llm.py file

CHROMA_PATH = os.getenv('CHROMA_PATH', './chroma_db')
COLLECTION_NAME = 'documents'

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
)

uploaded_files = {}


def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

def get_reranker(base_retriever):
    cohere_rerank = CohereRerank(
        cohere_api_key=os.getenv('CO_API_KEY'),
        model="rerank-english-v3.0",
        top_n=3
    )
    return ContextualCompressionRetriever(
        base_compressor=cohere_rerank,
        base_retriever=base_retriever
    )


def build_retriever_for_user(user_id):
    vectorstore = get_vectorstore()
    user_retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 6,
            "filter": {"user_id": user_id}
        }
    )


    user_chunks = []
    for chunks in uploaded_files.get(user_id, {}).values():
        user_chunks.extend(chunks)

    if user_chunks:
        docs = [Document(page_content=chunk) for chunk in user_chunks]
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 6
        ensemble = EnsembleRetriever(
            retrievers=[user_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )
        return get_reranker(ensemble)
    else:
        return get_reranker(user_retriever)


def load_vectorstore(filepath, user_id):
    filename = os.path.basename(filepath)

    if filepath.endswith('.pdf'):
        loader = PyPDFLoader(filepath)
        documents = loader.load()
        chunks_docs = splitter.split_documents(documents)
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        chunks_docs = splitter.create_documents([raw_text],
                                                metadatas=[{"source": filepath}])

    # tag every chunk with the filename so we can delete by source later
    for doc in chunks_docs:
        doc.metadata["source"] = filepath
        doc.metadata["user_id"] = user_id

        # store chunks per user
    if user_id not in uploaded_files:
        uploaded_files[user_id] = {}
    uploaded_files[user_id][filename] = [doc.page_content for doc in chunks_docs]

    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks_docs)

    print(f"Loaded {filename} for user {user_id} ({len(chunks_docs)} chunks)")


def delete_file(filename, user_id):
    user_files = uploaded_files.get(user_id, {})

    if filename not in user_files:
        return False

    # delete chunks from Chroma by source metadata
    vectorstore = get_vectorstore()

    all_results = vectorstore._collection.get(include=["metadatas"])
    ids_to_delete = [
        id_ for id_, meta in zip(all_results["ids"], all_results["metadatas"])
        if meta.get("user_id") == user_id
           and os.path.basename(meta.get("source", "")) == filename
    ]
    if ids_to_delete:
        vectorstore._collection.delete(ids=ids_to_delete)
        print(f"Deleted {len(ids_to_delete)} chunks for {filename} (user {user_id})")

    del uploaded_files[user_id][filename]
    return True

def get_uploaded_files(user_id):
    return list(uploaded_files.get(user_id, {}).keys())


def search(query, user_id):
    vectorstore = get_vectorstore()
    all_results = vectorstore._collection.get(include=["metadatas"])
    user_has_chunks = any(
        meta.get("user_id") == user_id
        for meta in all_results["metadatas"]
    )

    if not user_has_chunks:
        return []

    retriever = build_retriever_for_user(user_id)
    docs = retriever.invoke(query)

    return [
        {
            "content": doc.page_content,
            "page": doc.metadata.get("page", None),
            "source": doc.metadata.get("source", "Unknown")
        }
        for doc in docs
    ]