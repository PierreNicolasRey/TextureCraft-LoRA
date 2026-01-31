from pydantic import BaseModel
from typing import Literal

ModelVersion = Literal["final", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]

class PromptRequest(BaseModel):
    """Schéma des données reçues du backend Java."""
    prompt: str
    model_version: ModelVersion = "final"
    resolution: str
    negative_prompt: str | None = None
    seed: int | None = None

class ErrorDetails(BaseModel):
    status: str = "error"
    message: str
    code: int | None = None

class SuccessResponse(BaseModel):
    imageBase64Cible: str
    imageBase64Affichage: str
    model: str