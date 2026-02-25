import torch

# --- Chemins et ID de Modèle ---
LORA_ROOT_DIR = "src/resources/loras"
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Définition des versions/checkpoints LoRA disponibles
# Clé: Nom d'API (utilisé par Java), Valeur: Chemin relatif au LORA_ROOT_DIR
# Va évoluer une fois les itérations d'entrainement terminées
MODEL_VERSIONS = {
    "final": "final/pytorch_lora_weights.safetensors",
    "1": "checkpoint-200/pytorch_lora_weights.safetensors",
    "2": "checkpoint-400/pytorch_lora_weights.safetensors",
    "3": "checkpoint-600/pytorch_lora_weights.safetensors",
    "4": "checkpoint-800/pytorch_lora_weights.safetensors",
    "5": "checkpoint-1000/pytorch_lora_weights.safetensors",
    "6": "checkpoint-1200/pytorch_lora_weights.safetensors",
    "7": "checkpoint-1400/pytorch_lora_weights.safetensors",
    "8": "checkpoint-1600/pytorch_lora_weights.safetensors",
    "9": "checkpoint-1800/pytorch_lora_weights.safetensors",
    "10": "checkpoint-2000/pytorch_lora_weights.safetensors",
    "11": "checkpoint-2200/pytorch_lora_weights.safetensors",
    "12": "checkpoint-2400/pytorch_lora_weights.safetensors",
    "13": "checkpoint-2600/pytorch_lora_weights.safetensors",
    "14": "checkpoint-2800/pytorch_lora_weights.safetensors",
    "15": "checkpoint-3000/pytorch_lora_weights.safetensors",
    "16": "checkpoint-3200/pytorch_lora_weights.safetensors",
    "17": "checkpoint-3400/pytorch_lora_weights.safetensors",
    "18": "checkpoint-3600/pytorch_lora_weights.safetensors",
    "19": "checkpoint-3800/pytorch_lora_weights.safetensors",
    "20": "checkpoint-4000/pytorch_lora_weights.safetensors",
    "21": "checkpoint-4200/pytorch_lora_weights.safetensors",
    "22": "checkpoint-4400/pytorch_lora_weights.safetensors",
    "23": "checkpoint-4600/pytorch_lora_weights.safetensors",
    "24": "checkpoint-4800/pytorch_lora_weights.safetensors"
}

# --- Configuration Matérielle ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float32

# --- Paramètres de Génération par défaut ---
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_SEED = 6224
DEFAULT_NEG_PROMPT = "deformed, blurry, dull colors"

CHROMA_KEY_COLOR = (0, 255, 0)
RGB_TOLERANCE = 3
COLOR_TARGETS = {
    "black": (24, 24, 24),
    "blue": (51, 75, 178),
    "brown": (102, 77, 53),
    "cyan": (74, 128, 152),
    "glass": (139, 193, 205),
    "gray": (77, 77, 76),
    "green": (102, 129, 50),
    "light blue": (102, 153, 215),
    "light gray": (153, 153, 152),
    "orange": (216, 127, 53),
    "pink": (242, 128, 165),
    "purple": (127, 63, 179),
    "red": (153, 52, 53),
    "dark gray": (54, 40, 58),
    "white": (255, 255, 255),
    "yellow": (231, 230, 51),
    "lime": (127, 202, 25),
    "magenta": (178, 77, 216)
}