from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()



llm = Groq(
    api_key=os.getenv('GROQ_API_KEY'),
)


embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv('EMBEDDING_MODEL')
)