from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ModelType(str, Enum):
    team = "team"
    player = "player"


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message for the NBA analyst agent")
    max_iterations: int = Field(default=25, ge=1, le=50)


class ChatResponse(BaseModel):
    response: str


class TrainRequest(BaseModel):
    model_type: ModelType = Field(..., description="team or player XGBoost model")


class TrainResponse(BaseModel):
    model_type: ModelType
    message: str
    success: bool


class TeamPredictRequest(BaseModel):
    home_team: str = Field(..., min_length=1)
    away_team: str = Field(..., min_length=1)


class PlayerPredictRequest(BaseModel):
    player_a: str = Field(..., min_length=1)
    player_b: str = Field(..., min_length=1)


class PredictResponse(BaseModel):
    message: str
    success: bool


class ModelStatus(BaseModel):
    model_type: ModelType
    trained: bool
    model_path: str | None = None
    metrics: dict[str, Any] | None = None


class ModelsListResponse(BaseModel):
    models: list[ModelStatus]


class HealthResponse(BaseModel):
    status: str
    agent_ready: bool
    ollama_model: str
