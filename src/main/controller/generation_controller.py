from fastapi import APIRouter, HTTPException
from src.main.model.schemas import PromptRequest, SuccessResponse, ErrorDetails
from src.main.service import generation_service

router = APIRouter()

@router.post("/api/generer_texture", response_model=SuccessResponse)
def generate_image_endpoint(request: PromptRequest):
    """
    Endpoint principal pour générer une texture de bloc.
    Appelle le service de génération.
    """
    try:
        # Appelle le service pour obtenir l'image en Base64

        return generation_service.generer_texture(
            prompt=request.prompt,
            model_version=request.model_version,
            resolution_cible=request.resolution,
            negative_prompt=request.negative_prompt,
            seed=request.seed
        )
        
    except HTTPException as http_exc:
        # Cas 2: Erreur attendue (ex: modèle LoRA introuvable, model_version invalide)
        # On relève l'exception pour que FastAPI la gère et renvoie le statut correct (400, 500, etc.)
        _ = 1
        raise http_exc
        
    except Exception as e:
        # Cas 3: Erreur inattendue
        # HTTPException 500 : erreur serveur non gérée.
        print(f"Erreur interne inattendue : {e}")
        raise HTTPException(
            status_code=500,
            detail=ErrorDetails(
                status="error",
                message=f"Erreur interne du serveur: {type(e).__name__}",
                code=500 
            ).model_dump()
        )