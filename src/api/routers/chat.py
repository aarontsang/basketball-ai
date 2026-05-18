from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_agent
from api.schemas import ChatRequest, ChatResponse
from agent_service import chat

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def post_chat(body: ChatRequest) -> ChatResponse:
    try:
        agent = get_agent()
        response = await chat(agent, body.message, max_iterations=body.max_iterations)
        return ChatResponse(response=response)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc
