from fastapi import FastAPI
from src.main.controller import generation_controller
from src.main.service import generation_service

# Initialisation de l'application
app = FastAPI(
    title="TextureCraft LoRA API",
    description="API de génération de textures basée sur Stable Diffusion LoRA.",
    version="1.0.0"
)

# Charger le modèle de base au démarrage de l'application
# Si le chargement échoue, l'application s'arrête
generation_service.load_base_model()

# Ajouter les routes du contrôleur
app.include_router(generation_controller.router)

# Route de base pour vérifier que l'API est en ligne
@app.get("/")
def read_root():
    return {"status": "ok", "message": "API opérationnelle"}