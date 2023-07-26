from fastapi import Depends
from app.utils import AppModel
from ..service import Service, get_service
from . import router

# router = APIRouter()


class ChatRequest(AppModel):
    prompt: str


class ChatResponse(AppModel):
    response: str


@router.post("/")
def chat_with_ai(
    request: ChatRequest,
    svc: Service = Depends(get_service),
):
    prompt = request.prompt
    response = svc.chat_service.get_response(prompt)
    print(response)
    return response["content"]