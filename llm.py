from groq import Groq
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()



llm = Groq(
    api_key=os.getenv('GROQ_API_KEY'),
)


embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)