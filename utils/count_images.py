import os
from src.main.config.dotenv_config import get_env_path

# Adapter le chemin du répertoire dont on veut compter les images
ROOT_DIR = get_env_path("DIR_TO_COUNT_IMAGES")
counts = {}

# Parcourt les répertoires principaux
for dir_name in os.listdir(ROOT_DIR):
    dir_path = os.path.join(ROOT_DIR, dir_name)
    
    if os.path.isdir(dir_path):
        image_count = 0
        
        # Parcourt récursivement tous les sous-répertoires
        for root, _, files in os.walk(dir_path):
            for file in files:
                # Compte les images et leurs descriptions (on ne compte qu'un seul type)
                if file.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    image_count += 1
                    
        counts[dir_name] = image_count

# Affichage des résultats
total = 0
for category, count in sorted(counts.items()):
    total = total + count
    print(f"{category}: {count} images")

print(f"{total} images totales")