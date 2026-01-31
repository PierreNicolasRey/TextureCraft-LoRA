import os
import re
import sys
from typing import Union
from src.main.config.dotenv_config import get_env_path

ROOT_DIR = get_env_path("DIR_TO_RENAME")

def rename_folders_recursively(root_path: str, new_steps: int):
    """
    Parcourt récursivement le chemin root_path et renomme tous les sous-répertoires
    qui correspondent au format de concept <number>_<name>.
    
    Args:
        root_path (str): Le dossier parent à partir duquel commencer la recherche.
        new_steps (Union[int, str]): Le nouveau nombre de répétitions à utiliser.
    """
    
    # os.walk parcourt la hiérarchie: (dirpath, dirnames, filenames)
    # dirnames est la liste des dossiers dans dirpath
    
    # Nous commençons à partir du niveau ROOT_DIR et traitons tous les dossiers rencontrés
    # y compris ceux dans les sous-dossiers.
    
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
        
        for old_dir_name in dirnames:
            old_path = os.path.join(dirpath, old_dir_name)
            
            # Utiliser l'expression régulière pour détecter un préfixe existant (nombre + _)
            # Si le dossier est au format '10_concept' ou juste 'concept'
            match = re.match(r"^\d+_", old_dir_name)
            
            if match:
                # Un préfixe existe: Extrait le nom après le préfixe (ex: 'cobblestone')
                concept_name = old_dir_name[match.end():]
            elif old_dir_name.isdigit():
                 continue
            else:
                # Aucun préfixe n'existe: Le nom du concept est le nom entier du dossier
                concept_name = old_dir_name

            # Créer le nouveau nom de dossier
            new_dir_name = f"{new_steps}_{concept_name}"
            
            # Chemin complet du nouveau dossier
            new_path = os.path.join(dirpath, new_dir_name)
            
            # Renommer le dossier, mais seulement si le nom change
            if old_path != new_path:
                try:
                    os.rename(old_path, new_path)
                    print(f"Renommé : {os.path.relpath(old_path, root_path)} -> {os.path.relpath(new_path, root_path)}")
                except OSError as e:
                    print(f"Erreur lors du renommage de {old_dir_name} : {e}")


def main():
    # 1. Gestion de l'argument de la ligne de commande
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <nombre_de_repetitions>")
        print("Exemple: python changer_steps_nom_repertoires.py 3")
        sys.exit(1)
    
    try:
        steps = int(sys.argv[1])
        if steps <= 0:
            raise ValueError
    except ValueError:
        print("Erreur : Le nombre de répétitions doit être un entier positif.")
        sys.exit(1)
    
    # 2. Lancement de la fonction de renommage récursive
    print(f"Démarrage du renommage récursif pour le répertoire : {ROOT_DIR}")
    print(f"Nouvelles répétitions (steps) : {steps}")
    
    # Vérification initiale pour s'assurer que le chemin ROOT_DIR est valide
    if not os.path.isdir(ROOT_DIR):
        print(f"Erreur : Le répertoire de base est introuvable ou n'est pas un dossier : {ROOT_DIR}")
        sys.exit(1)
        
    rename_folders_recursively(ROOT_DIR, steps)
    print("Processus de renommage terminé.")


if __name__ == "__main__":
    main()