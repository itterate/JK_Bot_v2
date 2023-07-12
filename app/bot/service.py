import os

from dotenv import load_dotenv
from pydantic import BaseSettings

from .adapter.openai import ChatService

load_dotenv()  # Load environment variables from .env file


class Config(BaseSettings):
    HERE_API_KEY: str


class Service:
    def __init__(self, api_key: str):
        self.chat_service = ChatService(api_key)


def get_service():
    # print("token3")
    # print("TOKEN: ", os.getenv("OPENAI_API_KEY"))
    token = os.getenv("OPENAI_API_KEY")
    # print("token4")
    return Service(token)
