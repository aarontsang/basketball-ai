from fastapi import APIRouter

from api.deps import app_state
from api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        agent_ready=app_state.agent is not None,
        ollama_model=app_state.ollama_model,
    )
