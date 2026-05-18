import json
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from api.schemas import (
    ModelStatus,
    ModelType,
    ModelsListResponse,
    PlayerPredictRequest,
    PredictResponse,
    TeamPredictRequest,
    TrainRequest,
    TrainResponse,
)

router = APIRouter(prefix="/ml", tags=["ml"])

AGENT_CHATBOT_DIR = Path(__file__).resolve().parents[2] / "agent_chatbot"
if str(AGENT_CHATBOT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_CHATBOT_DIR))

from ml_training_handler import (  # noqa: E402
    predict_player_matchup,
    predict_team_matchup,
    train_player_xgboost_model,
    train_team_xgboost_model,
)
from training_service import metadata_path  # noqa: E402


def _is_success(message: str) -> bool:
    lowered = message.lower()
    return "failed" not in lowered and "error" not in lowered


@router.get("/models", response_model=ModelsListResponse)
async def list_models() -> ModelsListResponse:
    models: list[ModelStatus] = []
    for model_type in ModelType:
        meta_path = metadata_path(model_type.value)
        if meta_path.exists():
            data = json.loads(meta_path.read_text())
            models.append(
                ModelStatus(
                    model_type=model_type,
                    trained=True,
                    model_path=data.get("model_path"),
                    metrics=data.get("metrics"),
                )
            )
        else:
            models.append(ModelStatus(model_type=model_type, trained=False))
    return ModelsListResponse(models=models)


@router.post("/train", response_model=TrainResponse)
async def train_model(body: TrainRequest) -> TrainResponse:
    fn = (
        train_team_xgboost_model
        if body.model_type == ModelType.team
        else train_player_xgboost_model
    )
    message = await run_in_threadpool(fn)
    return TrainResponse(
        model_type=body.model_type,
        message=message,
        success=_is_success(message),
    )


@router.post("/predict/team", response_model=PredictResponse)
async def predict_team(body: TeamPredictRequest) -> PredictResponse:
    message = await run_in_threadpool(
        predict_team_matchup, body.home_team, body.away_team
    )
    if not _is_success(message):
        raise HTTPException(status_code=400, detail=message)
    return PredictResponse(message=message, success=True)


@router.post("/predict/player", response_model=PredictResponse)
async def predict_player(body: PlayerPredictRequest) -> PredictResponse:
    message = await run_in_threadpool(
        predict_player_matchup, body.player_a, body.player_b
    )
    if not _is_success(message):
        raise HTTPException(status_code=400, detail=message)
    return PredictResponse(message=message, success=True)
