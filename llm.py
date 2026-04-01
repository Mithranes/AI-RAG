from groq import Groq
from langchain_ollama.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()



llm = Groq(
    api_key=os.getenv('GROQ_API_KEY'),
)


embeddings = OllamaEmbeddings(
    model=os.getenv('EMBEDDING_MODEL'),
)