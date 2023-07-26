import os

from dotenv import load_dotenv

from .adapter.openai import ChatService

load_dotenv()  # Load environment variables from .env file


class Service:
    def __init__(self, aiapi_key: str, pinecone_key: str, pinecone_env: str):
        self.chat_service = ChatService(aiapi_key, pinecone_key, pinecone_env)
        

def get_service():
    openai_token = os.getenv("OPENAI_API_KEY")
    pinecone_token = os.getenv("PINECONE_API_KEY")
    pinecone_env_token = os.getenv("PINECONE_ENV")
    return Service(openai_token, pinecone_token, pinecone_env_token)
