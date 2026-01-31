import pytest
from starlette.testclient import TestClient
from src.main import main
import src.main.service.generation_service as generation_service
from unittest.mock import MagicMock
from fastapi import HTTPException

app = main.app

@pytest.fixture(autouse=True)
def mock_service_function(monkeypatch):
    """
    Mocke la fonction de service generer_texture pour isoler le test du contrôleur
    et garantir qu'elle retourne le modèle SuccessResponse attendu.
    """
    
    # Créer une instance MagicMock qui simule la réponse Pydantic attendue
    mock_success_response = MagicMock()
    # Utilisation de valeurs distinctes pour les assertions
    mock_success_response.imageBase64Cible = "MOCK_CIBLE_BASE64_DATA_A_BIT_LONG"
    mock_success_response.imageBase64Affichage = "MOCK_AFFICHAGE_BASE64_DATA"
    mock_success_response.model = "final"
    
    # 1. Mock de la fonction generer_texture elle-même (CLÉ pour l'isolation)
    mock_generer_texture = MagicMock(return_value=mock_success_response)
    
    # 2. Application du mock
    monkeypatch.setattr(generation_service, "generer_texture", mock_generer_texture)
    
    # 3. Mocks minimalistes pour éviter les erreurs d'initialisation (si le contrôleur les appelle)
    monkeypatch.setattr(generation_service, "DEVICE", "cpu") 
    monkeypatch.setattr(generation_service, "DTYPE", None) 
    monkeypatch.setattr(generation_service, "LORA_ROOT_DIR", "/mock/lora/root")
    monkeypatch.setattr(generation_service, "MODEL_VERSIONS", {"final": "mock.safetensors"})
    
    # On retourne le mock pour d'éventuelles assertions sur les arguments d'appel
    yield mock_generer_texture


@pytest.fixture(scope="module")
def client():  
    # Créer le client de test pour l'application
    with TestClient(app=app, base_url="http://test") as c:
        yield c

def test_api_root_status_ok(client):
    """Vérifie que l'API de base est en ligne."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_generate_success(client, mock_service_function):
    """Teste le flux complet de génération (succès 200)."""
    payload = {
        "prompt": "a minecraft block texture, high resolution",
        "model_version": "final",
        "resolution": "16x16" 
    }
    response = client.post("/api/generer_texture", json=payload)
    
    # Vérification que le service a été appelé avec les bons arguments
    mock_service_function.assert_called_once_with(
        prompt="a minecraft block texture, high resolution",
        model_version="final",
        resolution_cible="16x16",
        negative_prompt=None,
        seed=None
    )

    assert response.status_code == 200
    data = response.json()
    
    # Vérification que la réponse JSON est correctement sérialisée à partir du mock
    assert data["imageBase64Cible"] == "MOCK_CIBLE_BASE64_DATA_A_BIT_LONG"
    assert data["imageBase64Affichage"] == "MOCK_AFFICHAGE_BASE64_DATA"
    assert data["model"] == "final"


def test_service_error_422(client, mock_service_function):
    """Teste la gestion d'une HTTPException levée par le service (ex: erreur métier 422)."""
    # Configurer le mock du service pour lever une HTTPException 422
    mock_service_function.side_effect = HTTPException(status_code=422, detail="Version du modèle invalide.")
    
    payload = {
        "prompt": "test",
        "model_version": "version_inconnue" ,
        "resolution": "16x16" 
    }
    response = client.post("/api/generer_texture", json=payload)
    
    assert response.status_code == 422
    assert response.json()["detail"][0].get("msg") == "Input should be 'final', 'etape_880', 'etape_440' or 'etape_220'"

def test_service_error_500(client, mock_service_function):
    """Teste la gestion d'une exception non gérée (500)."""
    # Configurer le mock du service pour lever une erreur Python standard
    mock_service_function.side_effect = ValueError("Erreur de calcul mockée.")
    
    payload = {
        "prompt": "test",
        "model_version": "final" ,
        "resolution": "16x16" 
    }
    response = client.post("/api/generer_texture", json=payload)
    
    assert response.status_code == 500
    # Vérifie que la réponse suit le format ErrorDetails du 500
    assert "Erreur interne du serveur: ValueError" in response.json()["detail"]["message"]