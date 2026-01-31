import pytest
from unittest.mock import patch, MagicMock
import os
from fastapi import HTTPException
import torch
import src.main.service.generation_service as generationService
from src.main.model.schemas import SuccessResponse
from src.main.config.settings import CHROMA_KEY_COLOR, RGB_TOLERANCE, COLOR_TARGETS
from PIL import Image

# Nous devons ajuster les imports pour simuler les configurations
# Créez des mocks pour les configurations (settings) pour ne pas dépendre du fichier réel
# et simplifier l'exécution des tests.

# --- Mocks pour les Configurations et le Code de Service ---
# Simule les chemins d'accès et les valeurs par défaut
MOCK_LORA_ROOT = "/mock/lora/root"
MOCK_MODEL_VERSIONS = {
    "final": "weights_final.safetensors",
    "etape_440": "checkpoint-440/weights.safetensors",
}

ORIGINAL_ISINSTANCE = isinstance

# Créez un module mock pour les settings
class MockSettings:
    LORA_ROOT_DIR = MOCK_LORA_ROOT
    MODEL_VERSIONS = MOCK_MODEL_VERSIONS
    DEFAULT_NEG_PROMPT = "low quality"
    DEFAULT_GUIDANCE_SCALE = 7.5
    DEFAULT_STEPS = 30
    DEFAULT_SEED = 42
    DEVICE = "cpu"
    DTYPE = torch.float32

@pytest.fixture(autouse=True)
def reset_lora_actif():
    """Réinitialise l'état global lora_actif entre chaque test."""
    generationService.lora_actif = None
    yield

@pytest.fixture
def mock_service_dependencies(monkeypatch):
    """
    Patche les constantes et initialise l'objet pipe mocké.
    Cette version patche 'isinstance' pour contourner le conflit d'héritage de PIL.
    """
    
    # 1. Patchage des constantes (Settings)
    monkeypatch.setattr(generationService, "LORA_ROOT_DIR", MOCK_LORA_ROOT)
    monkeypatch.setattr(generationService, "MODEL_VERSIONS", MOCK_MODEL_VERSIONS)
    
    # --- Création d'un mock d'objet Image PIL qui passe l'instance check ---
    
    # 2. Création d'un MagicMock standard pour l'image
    mock_image_instance = MagicMock()
    
    # Simuler les attributs et méthodes nécessaires
    mock_image_instance.mode = "RGB"
    mock_image_instance.size = (512, 512)
    
    # 3. Simuler la méthode .save()
    def mock_image_save(buffer, format):
        buffer.write(b'MOCK_PNG_DATA_A_BIT_LONG')
        
    mock_image_instance.save.side_effect = mock_image_save
    
    # Simuler les méthodes de manipulation
    mock_image_instance.resize.return_value = mock_image_instance
    mock_image_instance.convert.return_value = mock_image_instance 
    mock_image_instance.getdata.return_value = [(0, 0, 0, 255)] * (512*512)
    mock_image_instance.split.return_value = [mock_image_instance] * 4

    # 4. Patch de la fonction 'isinstance'
    PIL_IMAGE_CLASS = Image.Image 

    def mock_isinstance(obj, classinfo):
        """Fonction mockée pour isinstance."""
        # Si la vérification porte sur la classe Image.Image ET l'objet est notre mock, on renvoie True
        if classinfo is PIL_IMAGE_CLASS and obj is mock_image_instance:
             return True
        # Sinon, on exécute la fonction isinstance originale
        return ORIGINAL_ISINSTANCE(obj, classinfo)

    # Appliquer le patch au built-in 'isinstance'
    monkeypatch.setattr('builtins.isinstance', mock_isinstance) 

    # --- Mocks pour le Pipeline ---
    mock_pipe_instance = MagicMock()
    
    # Le call du pipe retourne une liste contenant notre mock d'Image
    mock_pipe_call = MagicMock(images=[mock_image_instance])
    mock_pipe_instance.return_value = mock_pipe_call
    
    # 5. Mettre le mock en place dans le module de service
    # (J'ai conservé "pipe" selon votre dernière structure)
    monkeypatch.setattr(generationService, "pipe", mock_pipe_instance)
    
    # 6. Retourner l'instance du mock pipe pour les assertions
    return mock_pipe_instance

# --- TESTS METHODE load_base_model ---

@patch("src.main.service.generation_service.StableDiffusionPipeline")
def test_load_base_model_success(MockSDP):
    """Vérifie le chargement initial du modèle."""
    # S'assurer que le pipe est None avant de commencer
    generationService.pipe = None 
    
    # Charger le modèle
    generationService.load_base_model()
    
    # Vérifications:
    MockSDP.from_pretrained.assert_called_once()
    assert generationService.pipe is not None
    generationService.load_base_model()
    MockSDP.from_pretrained.assert_called_once()

# --- TESTS METHODE generer_texture  ---

def test_generer_texture_no_base_model():
    """Teste l'échec si le modèle de base n'est pas chargé (pipe is None)."""
    generationService.pipe = None
    
    with pytest.raises(HTTPException) as excinfo:
        generationService.generer_texture("prompt", "final", "16x16")
        
    assert excinfo.value.status_code == 500
    assert "n'est pas chargé" in excinfo.value.detail

def test_generer_texture_invalid_version():
    """Teste l'échec si la version du modèle n'existe pas dans la config (Erreur 400)."""
    # Pipe doit être défini pour que le test atteigne la validation de version
    generationService.pipe = MagicMock() 
    
    with pytest.raises(HTTPException) as excinfo:
        generationService.generer_texture("prompt", "version_inconnue", "16x16")
        
    assert excinfo.value.status_code == 400
    assert "invalide" in excinfo.value.detail

@patch("os.path.exists", return_value=False)
def test_generer_texture_lora_file_not_found(mock_exists):
    """Teste l'échec si le fichier LoRA n'est pas trouvé (Erreur 500)."""
    generationService.pipe = MagicMock()
    
    with pytest.raises(HTTPException) as excinfo:
        generationService.generer_texture("prompt", "final", "16x16")
        
    assert excinfo.value.status_code == 500
    assert "introuvable" in excinfo.value.detail

@patch("os.path.exists", return_value=True)
def test_generer_texture_lora_load_failure(mock_exists):
    """Teste l'échec si le chargement du LoRA plante (Erreur 500)."""
    # Configurer le mock pipe pour lever une exception lors du load_attn_procs
    mock_pipe = MagicMock()
    mock_pipe.load_lora_weights.side_effect = Exception("Erreur PyTorch simulée")
    generationService.pipe = mock_pipe
    
    with pytest.raises(HTTPException) as excinfo:
        generationService.generer_texture("prompt", "final", "16x16")
        
    assert excinfo.value.status_code == 500
    assert "chargement des poids LoRA" in excinfo.value.detail
    assert generationService.lora_actif is None

# --- Test de Succès ---
@patch("src.main.service.generation_service.BytesIO")
@patch("os.path.exists", return_value=True)
def test_generer_texture_success_first_call(mock_exists, MockBytesIO, mock_service_dependencies):
    """Teste le flux complet de succès, premier appel (Doit charger le LoRA)."""
    mock_bytesio_instance = MockBytesIO.return_value
    mock_bytesio_instance.getvalue.return_value = b'MOCK_PNG_DATA_A_BIT_LONGER_THAN_10_BYTES'
    mock_bytesio_instance.write.return_value = None
    
    assert generationService.lora_actif is None

    response = generationService.generer_texture("test prompt", "final","16x16", seed=100)

    # 1. Vérification du chargement
    expected_path = os.path.join(MOCK_LORA_ROOT, MOCK_MODEL_VERSIONS["final"])
    mock_service_dependencies.load_lora_weights.assert_called_with(expected_path)
    # 2. Vérification de l'état
    assert generationService.lora_actif == "final"
    
    # 3. Le pipeline de génération a été appelé
    generationService.pipe.assert_called_once()
    assert isinstance(response, SuccessResponse)
    assert len(response.imageBase64Affichage) > 10
    
@patch("src.main.service.generation_service.BytesIO")
@patch("os.path.exists", return_value=True)
def test_generer_texture_success_cached_call(mock_exists, MockBytesIO, mock_service_dependencies):
    """Teste le flux de succès, LoRA déjà chargé (NE DOIT PAS recharger)."""
    mock_bytesio_instance = MockBytesIO.return_value
    mock_bytesio_instance.getvalue.return_value = b'MOCK_PNG_DATA_A_BIT_LONGER_THAN_10_BYTES'
    mock_bytesio_instance.write.return_value = None

    generationService.lora_actif = "final"

    mock_service_dependencies.load_lora_weights.reset_mock()
    generationService.pipe.reset_mock()
    
    generationService.generer_texture("test prompt", "final", "16x16", seed=100)

    # 1. Vérification du chargement
    mock_service_dependencies.load_lora_weights.assert_not_called()
    
    # 2. Vérification de l'état
    assert generationService.lora_actif == "final"
    
    # 3. La génération a bien été appelée
    generationService.pipe.assert_called_once()


@patch("src.main.service.generation_service.BytesIO")
@patch("os.path.exists", return_value=True)
def test_generer_texture_success_switch_call(mock_exists, MockBytesIO, mock_service_dependencies):
    """Teste le flux de succès lors du changement de modèle LoRA (Doit recharger)."""
    mock_bytesio_instance = MockBytesIO.return_value
    mock_bytesio_instance.getvalue.return_value = b'MOCK_PNG_DATA_A_BIT_LONGER_THAN_10_BYTES'
    mock_bytesio_instance.write.return_value = None

    generationService.lora_actif = "etape_440"

    mock_service_dependencies.load_lora_weights.reset_mock()
    generationService.pipe.reset_mock()
    
    generationService.generer_texture("test prompt", "final", "16x16", seed=100)

    # 1. Vérification du chargement
    expected_path = os.path.join(MOCK_LORA_ROOT, MOCK_MODEL_VERSIONS["final"])
    mock_service_dependencies.load_lora_weights.assert_called_once_with(expected_path)
    
    # 2. Vérification de l'état
    assert generationService.lora_actif == "final"
    
    # 3. La génération a bien été appelée
    generationService.pipe.assert_called_once()


# --- TESTS METHODE extract_prompt_info ---

def test_extract_prompt_infos_simple_image():
    mockPrompt = "color : blue"

    response = generationService.extract_prompt_infos(mockPrompt)

    assert response.get("has_background_tag") == False
    assert response.get("is_opacity_transparent") == False
    assert response.get("is_opacity_semi_transparent") == False
    assert response.get("colors")[0] == "blue"

def test_extract_prompt_infos_non_full_image():
    mockPrompt = "color : blue, background : transparent"

    response = generationService.extract_prompt_infos(mockPrompt)

    assert response.get("has_background_tag") == True
    assert response.get("is_opacity_transparent") == False
    assert response.get("is_opacity_semi_transparent") == False
    assert response.get("colors")[0] == "blue"

def test_extract_prompt_infos_full_transparent_image():
    mockPrompt = "color : blue, opacity : transparent"

    response = generationService.extract_prompt_infos(mockPrompt)

    assert response.get("has_background_tag") == False
    assert response.get("is_opacity_transparent") == True
    assert response.get("is_opacity_semi_transparent") == False
    assert response.get("colors")[0] == "blue"

def test_extract_prompt_infos_semi_transparent_image():
    mockPrompt = "color : blue, opacity : semi-transparent"

    response = generationService.extract_prompt_infos(mockPrompt)

    assert response.get("has_background_tag") == False
    assert response.get("is_opacity_transparent") == False
    assert response.get("is_opacity_semi_transparent") == True
    assert response.get("colors")[0] == "blue"

def test_extract_prompt_infos_full_transparent_and_non_full_image():
    mockPrompt = "color : blue, background : transparent, opacity : transparent"

    response = generationService.extract_prompt_infos(mockPrompt)

    assert response.get("has_background_tag") == True
    assert response.get("is_opacity_transparent") == True
    assert response.get("is_opacity_semi_transparent") == False
    assert response.get("colors")[0] == "blue"

def test_find_closest_color_target_with_color_in_dict():
    color_names = ["blue", "red"]

    response = generationService.find_closest_color_target(60, 70, 180, color_names)

    assert response == "blue"

def test_find_closest_color_target_with_color_not_in_dict():
    color_names = ["indigo", "fushia"]

    response = generationService.find_closest_color_target(60, 70, 180, color_names)

    assert response == None


# --- TESTS METHODE redimensionner_texture --- 
def test_redimensionner_texture_OK():
    inputImage: Image = Image.new("RGB", (512,512), "red")
    expectedImage: Image = Image.new("RGB", (16,16), "red")

    resolution: str = "16x16"

    returnedImage = generationService.redimensionner_texture(inputImage, resolution)

    assert returnedImage.size == expectedImage.size
    assert returnedImage.mode == inputImage.mode

def test_redimensionner_texture_KO_unexpected_resolution():
    inputImage: Image = Image.new("RGB", (512,512), "red")

    resolution: str = "mauvaise résolution"

    with pytest.raises(HTTPException) as exc_info:
        generationService.redimensionner_texture(inputImage, resolution)
    
    assert exc_info.type is HTTPException
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Format de résolution invalide ou valeurs non positives. Attendu : 'WxH' (W, H > 0)."

def test_redimensionner_texture_KO_resolution_should_be_positive():
    inputImage: Image = Image.new("RGB", (512,512), "red")

    resolution: str = "0x0"

    with pytest.raises(HTTPException) as exc_info:
        generationService.redimensionner_texture(inputImage, resolution)
    
    assert exc_info.type is HTTPException
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "La résolution cible doit être positive et non nulle."


# --- TESTS METHODE apply_chroma_key
@patch('src.main.service.generation_service.extract_prompt_infos')
def test_apply_chroma_key_binary_success_with_green(mock_extract):
    """
    Teste le scénario binaire : Le Vert Pur doit devenir transparent (Alpha=0),
    et un Bleu non chromé doit rester opaque (Alpha=255).
    """
    # 1. Préparation des Mocks : Pas de transparence demandée, entre dans le ELSE
    mock_extract.return_value = {
        'is_opacity_transparent': False, 
        'is_opacity_semi_transparent': False,
        'colors': []
    }
    
    # Pixel 1: Fond (Vert Pur), dans la tolérance
    green_key_pixel = (0, 255, 0)
    
    # Pixel 2: Contenu (Bleu Pur), hors tolérance
    blue_pixel = (0, 0, 255)
    
    # Créer l'image 2x1
    input_image = create_pixel_image(green_key_pixel, size=(2, 1))
    input_image.putdata([green_key_pixel, blue_pixel]) 
    
    # 2. Exécution
    result_image = generationService.apply_chroma_key(input_image, "prompt simple") 
    
    # 3. Assertions
    assert result_image.mode == "RGBA"
    data = list(result_image.getdata())
    
    # Pixel 1 (Vert) : Doit être transparent (Alpha=0)
    # NOTE: La couleur RGB reste, seul l'Alpha change
    assert data[0] == (0, 255, 0, 0) 
    
    # Pixel 2 (Bleu) : Doit être opaque (Alpha=255)
    assert data[1] == (0, 0, 255, 255)

@patch('src.main.service.generation_service.extract_prompt_infos')
def test_apply_chroma_key_opacity_transparent(mock_extract):
    # 1. Préparation des Mocks : Pas de transparence demandée, entre dans le ELSE
    mock_extract.return_value = {
        'has_background_transparent': False,
        'is_opacity_transparent': True, 
        'is_opacity_semi_transparent': False,
        'colors': ['red']
    }

    image_rgb: Image = Image.new("RGB", (16,16), (153, 203, 53))
    expected_alpha = 65

    image_rgba: Image = generationService.apply_chroma_key(image_rgb, "prompt simple")

    assert image_rgba.getdata()[0] == (153, 52, 53, expected_alpha)

@patch('src.main.service.generation_service.extract_prompt_infos')
def test_apply_chroma_key_opacity_color_not_found(mock_extract):
    # 1. Préparation des Mocks : Pas de transparence demandée, entre dans le ELSE
    mock_extract.return_value = {
        'has_background_transparent': False,
        'is_opacity_transparent': False, 
        'is_opacity_semi_transparent': True,
        'colors': None
    }

    image_rgb: Image = Image.new("RGB", (16,16), (153, 203, 53))
    expected_alpha = 255

    image_rgba: Image = generationService.apply_chroma_key(image_rgb, "prompt simple")

    assert image_rgba.getdata()[0] == (153, 203, 53, expected_alpha)


# --- Fonctions Utilitaires pour les Tests ---

def create_pixel_image(color_rgb, size=(1, 1)):
    """Crée une image avec une couleur spécifique (R, G, B)."""
    img = Image.new("RGB", size)
    # Convertir le tuple de couleur en tuple PIL (nécessaire si on passe des variables)
    img.putdata([color_rgb] * size[0] * size[1])
    return img