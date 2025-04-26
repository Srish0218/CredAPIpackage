from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GEMINI_API"))
timely_closing_ST_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
