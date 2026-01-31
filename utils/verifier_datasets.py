from pathlib import Path
from src.main.config.dotenv_config import get_env_path

# --- CONFIGURATION ---
TRAIN_IMAGE_ROOT = Path(get_env_path("DIR_TRAIN_DATASET_TO_VERIFY"))
CONTROL_IMAGE_ROOT = Path(get_env_path("DIR_CONTROL_DATASET_TO_VERIFY"))
IMAGE_EXTENSIONS = ('.png')
# ---------------------

def get_caption_root(image_root: Path) -> Path:
    """Déduit le chemin racine des fichiers de description (.txt) à partir du chemin racine des images."""
    # Remplace 'images' par 'captions' dans le chemin de manière sécurisée
    path_parts = list(image_root.parts)
    try:
        images_index = path_parts.index('images')
        path_parts[images_index] = 'captions'
        return Path(*path_parts)
    except ValueError:
        # Cas où 'images' n'est pas dans le chemin
        raise ValueError(f"Le chemin d'image '{image_root}' ne contient pas 'images', impossible de déduire le chemin de description.")

def verify_captions(image_root: Path, is_control: bool = False):
    """Vérifie que chaque image dans image_root a son fichier .txt correspondant dans le répertoire déduit."""
    
    caption_root = get_caption_root(image_root)
    dataset_name = image_root.parts[-3] if len(image_root.parts) >= 3 else image_root.name
    
    print(f"--- Vérification des Captions pour {dataset_name} ({'Control' if is_control else 'Train'}) ---")
    print(f"Images: {image_root}")
    print(f"Captions attendues dans: {caption_root}")
    
    missing_captions = []
    
    # Parcourt tous les fichiers images de manière récursive
    for image_path in image_root.glob('**/*'):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            
            # 1. Détermine le chemin relatif (ex: 'wood/oak/oak_log.png')
            relative_path = image_path.relative_to(image_root)
            
            # 2. Construit le chemin absolu du .txt attendu
            caption_file = caption_root / relative_path.with_suffix('.txt')
            
            if not caption_file.exists():
                missing_captions.append(relative_path)
                
    if missing_captions:
        print(f"ERREUR: {len(missing_captions)} images SANS fichier de description (.txt) correspondant trouvées:")
        for missing in missing_captions:
            print(f"  - {missing.with_suffix('.txt')}")
        return False
    else:
        print("Tous les fichiers images ont leur fichier de description (.txt) correspondant.")
        return True

def main():
    try:
        caption_ok_train = verify_captions(TRAIN_IMAGE_ROOT)
        caption_ok_control = verify_captions(CONTROL_IMAGE_ROOT, is_control=True)
        
        if not (caption_ok_train and caption_ok_control):
            print("Correction des fichiers de description requise.")
            
    except ValueError as e:
        print(f"ERREUR DE CONFIGURATION: {e}")

if __name__ == "__main__":
    main()