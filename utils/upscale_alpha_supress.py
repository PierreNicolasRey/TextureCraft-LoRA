from pathlib import Path
from PIL import Image
import shutil
from src.main.config.dotenv_config import get_env_path


# --- CONFIGURATION ---
SOURCE_IMAGE_ROOT = Path(get_env_path("DIR_TRAIN_DATASET_TO_UPSCALE"))
SOURCE_CONTROL_ROOT = Path(get_env_path("DIR_CONTROL_DATASET_TO_UPSCALE"))
SOURCE_DIRS = [
    SOURCE_IMAGE_ROOT,
    SOURCE_CONTROL_ROOT
]

# NOUVEAUX CHEMINS POUR LES DONNÉES UPSCALÉES (créer ces dossiers manuellement)
UPSCALED_IMAGE_ROOT = Path(get_env_path("DIR_TRAIN_DATASET_UPSCALED"))
UPSCALED_CONTROL_ROOT = Path(get_env_path("DIR_CONTROL_DATASET_UPSCALED"))
UPSCALED_DIRS = [
    UPSCALED_IMAGE_ROOT,
    UPSCALED_CONTROL_ROOT
]

IMAGE_EXTENSIONS = ('.png')
CAPTION_EXTENSION = '.txt'
RESOLUTION = (512, 512)
# Couleur clé: Vert pur (0, 255, 0) - doit être une couleur absente des textures
CHROMA_KEY_COLOR = (0, 255, 0)
# ---------------------

def process_directory(source_dir: Path, target_dir: Path):
    """Parcourt récursivement le répertoire source, applique les transformations aux images
    et copie les fichiers de caption dans la structure cible.
    """
    if not source_dir.is_dir():
        print(f"ATTENTION: Le répertoire source {source_dir} n'existe pas.")
        return
    
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Traitement de {source_dir} vers {target_dir} ---")
    
    count = 0
    
    for file_path in source_dir.rglob('*'):
        relative_path = file_path.relative_to(source_dir)
        target_file_path = target_dir / relative_path

        # Si c'est un répertoire, on le crée et on passe à l'élément suivant
        if file_path.is_dir():
            target_file_path.mkdir(parents=True, exist_ok=True)
            continue
            
        # --- 1. GESTION DES CAPTIONS (.txt) ---
        if file_path.suffix.lower() == CAPTION_EXTENSION:
            if copy_file(file_path, target_file_path):
                count += 1
            continue

        # --- 2. GESTION DES IMAGES (.png) ---
        if file_path.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                original_image = Image.open(file_path)
                
                # Appliquer la logique de transformation (Upscale + Chroma Key)
                image_chroma = suppress_alpha(original_image, CHROMA_KEY_COLOR)
                final_image = upscale_image(image_chroma, RESOLUTION)
                
                # Sauvegarde
                final_image.save(target_file_path, format='PNG')
                count += 1
                
            except Exception as e:
                print(f"ERREUR lors du traitement de l'image {file_path.name}: {e}")
                
    print(f"--- {count} fichiers (images et captions) traités dans {source_dir.name}. ---")

def upscale_image(image: Image, target_size: tuple[int, int]) -> Image:
    """
    Redimensionne l'image à la résolution cible en utilisant l'algorithme 
    Nearest Neighbor pour préserver la netteté du pixel art.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("L'input n'est pas un objet PIL Image.")
    
    resized_image = image.resize(
        target_size, 
        resample=Image.Resampling.NEAREST
    )
    return resized_image

def suppress_alpha(image: Image, chroma_key_color: tuple[int, int, int]) -> Image:
    """
    Remplace la transparence (canal alpha) de l'image par une couleur clé (chroma key).
    """

    # 1. Normalisation en RGBA pour garantir l'accès à l'alpha si elle existe
    image_rgba = image.convert('RGBA') 
    
    # 2. Le reste de la logique pour la suppression de l'alpha
    *_, alpha = image_rgba.split() 
    rgb_image = Image.new("RGB", image_rgba.size, chroma_key_color)
    rgb_image.paste(image_rgba, mask=alpha)
    
    return rgb_image

def copy_file(source_path: Path, target_path: Path) -> bool:
    """
    Copie un fichier source vers un chemin cible.
    Retourne True si la copie réussit, False sinon.
    """
    try:
        # S'assurez que les répertoires parents existent si la cible est un fichier
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        return True
    except Exception as e:
        print(f"ERREUR lors de la copie du fichier {source_path.name}: {e}")
        return False
    
def main():
    try:
        for root, out in zip(SOURCE_DIRS, UPSCALED_DIRS):
            process_directory(root, out)
            
    except Exception as e:
        print(f"ERREUR FATALE: {e}")

if __name__ == "__main__":
    main()