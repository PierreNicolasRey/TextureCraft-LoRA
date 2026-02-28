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

    if source_path.resolve() == Path.cwd().resolve():
        print("Sécurité : La source pointe sur la racine du projet. Annulation.")
        return

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

    print(f"\nDébut du nettoyage")
    for item in os.listdir(source_path):
        item_path = source_path / item
        try:
            if item_path.is_dir():
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
            print(f"\nFin du nettoyage")
        except Exception as e:
            print(f"Impossible de supprimer {item}: {e}")

if __name__ == "__main__":
    sync_lora_models_smart(SOURCE_LORA_DIR, LORA_ROOT_DIR)