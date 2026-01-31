# Script parcourant le fichier captions.json pour créer les descriptions 
# associées à chaque image du dataset en suivant l'arboresence de ces images

import os
import json

# --- Définition de la fonction pour parcourir le JSON ---
def show_indices(obj, indices):
    """
    Parcourt récursivement un objet (dict ou list) et renvoie les chemins de clés et les valeurs finales.
    Chaque chemin de clé est une liste d'indices/clés.
    """
    for k, v in obj.items() if isinstance(obj, dict) else enumerate(obj):
        if isinstance(v, (dict, list)):
            yield from show_indices(v, indices + [k])
        else:
            yield indices + [k], v

# --- Nom du fichier JSON et du répertoire de base ---
JSON_FILE = 'captions.json'
BASE_DIR = 'dataset_block_control/captions' # Le répertoire racine où tout sera créé (répertoire courant)

# --- 1. S'assurer que le répertoire de base existe ---
os.makedirs(BASE_DIR, exist_ok=True) 

# --- 2. Charger les données JSON ---
try:
    with open(JSON_FILE, 'r', encoding='utf-8') as file:
        data = json.load(file)
except FileNotFoundError:
    print(f"Erreur: Le fichier '{JSON_FILE}' n'a pas été trouvé.")
    exit()
except json.JSONDecodeError:
    print(f"Erreur: Impossible de décoder le fichier '{JSON_FILE}'. Vérifiez son format JSON.")
    exit()

# --- 3. Parcourir les données et créer les structures ---
print(f"Début du traitement des données et création des structures dans '{BASE_DIR}'...")

for keys, value in show_indices(data, []):
    # La dernière clé (keys[-1]) est le nom du fichier (ex: 'amethyst_block.txt')
    # Les clés précédentes (keys[:-1]) sont les noms des répertoires parents
    
    # 3.1. Construire le chemin du répertoire
    # On ajoute le répertoire de base et on joint toutes les clés de répertoire
    dir_parts = [BASE_DIR] + keys[:-1]
    output_dir = os.path.join(*dir_parts)
    
    # 3.2. Construire le chemin complet du fichier
    filename = keys[-1]
    output_filepath = os.path.join(output_dir, filename)
    
    # 3.3. Créer les répertoires si nécessaire
    # os.makedirs(..., exist_ok=True) crée tous les répertoires parents nécessaires 
    # et ignore l'erreur si le répertoire existe déjà.
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 3.4. Écrire la valeur dans le fichier
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(str(value)) # On s'assure que la valeur est écrite comme une chaîne
            
        print(f"Créé: {output_filepath}")
        
    except Exception as e:
        print(f"Erreur lors de la création de '{output_filepath}': {e}")

print("Traitement terminé.")
