from groq import Groq
from langchain_community.embeddings import FastEmbedEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()



llm = Groq(
    api_key=os.getenv('GROQ_API_KEY'),
)


embeddings = FastEmbedEmbeddings(
    model_name=os.getenv('EMBEDDING_MODEL')
)
