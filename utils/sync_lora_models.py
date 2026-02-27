import os
import shutil
from pathlib import Path
from src.main.config.settings import LORA_ROOT_DIR
from src.main.config.dotenv_config import get_env_path

SOURCE_LORA_DIR = Path(get_env_path("SOURCE_LORA_DIR"))

def sync_lora_models_smart(source_path, app_resources_path):
    source_path = Path(source_path)
    app_resources_path = Path(app_resources_path)
    app_resources_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Vérification des nouveaux modèles dans {source_path}")

    if not source_path.exists():
        print(f"Erreur : Chemin source introuvable.")
        return

    items = os.listdir(source_path)
    checkpoint_dirs = [d for d in items if (d.startswith("checkpoint-") or d.startswith("checkpoint_")) and (source_path / d).is_dir()]

    for folder_name in checkpoint_dirs:
        source_folder = source_path / folder_name
        dest_folder = app_resources_path / folder_name
        
        safetensors_files = [f for f in os.listdir(source_folder) if f.endswith(".safetensors")]
        
        if not safetensors_files:
            continue
            
        source_file = source_folder / safetensors_files[0]
        
        dest_folder.mkdir(exist_ok=True)
        dest_file = dest_folder / safetensors_files[0]

        if dest_file.exists():
            continue
        
        print(f"Nouveau modèle trouvé : {folder_name}. Copie en cours.")
        shutil.copy2(source_file, dest_file)

    print(f"\nFin de la synchronisation")

    try:
        print(f"Nettoyage du dossier source : {source_path}")
        shutil.rmtree(source_path)
        source_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Erreur lors du nettoyage : {e}")

sync_lora_models_smart(SOURCE_LORA_DIR, LORA_ROOT_DIR)