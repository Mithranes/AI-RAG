from groq import Groq
from langchain_community.embeddings import FastEmbedEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()



llm = Groq(
    api_key=os.getenv('GROQ_API_KEY'),
)


embeddings = OpenAIEmbeddings(
    api_key=os.getenv('OPENAI_API_KEY'),
    model=os.getenv('EMBEDDING_MODEL')
)
