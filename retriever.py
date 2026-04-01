from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from llm import embeddings
import os


FAISS_PATH = 'faiss_index'

# global retriever so it can be reloaded anytime
retriever = None

def load_vectorstore(filepath):
    global retriever

    if filepath.endswith('.pdf'):
        loader = PyPDFLoader(filepath)
        documents = loader.load()
        raw_text = "\n".join([doc.page_content for doc in documents])
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = splitter.split_text(raw_text)

    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vectorstore.save_local(FAISS_PATH)
    print(f'✅ Vectorstore saved to {FAISS_PATH}')

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print(f"✅ Vectorstore loaded from {filepath}")


def load_saved_vectorstore():
    global retriever
    if os.path.exists(FAISS_PATH):
        vectorstore = FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        print(f"✅ Vectorstore loaded from disk")


def search(query):
    if retriever is None:
        return ["No document loaded yet. Please upload a file first."]
    docs = retriever.invoke(query)
    return [doc.page_content for doc in docs]


load_saved_vectorstore()
