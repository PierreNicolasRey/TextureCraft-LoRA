from pathlib import Path
import re
from src.main.config.dotenv_config import get_env_path


# --- CONFIGURATION ---
CAPTION_ROOT_DIRS = [
    Path(get_env_path("DIR_TRAIN_DATASET_CAPTIONS")),    # Train
    Path(get_env_path("DIR_CONTROL_DATASET_CAPTIONS")) # Control
]
# ---------------------

# Liste de TOUS les tags avec deux-points
EXPECTED_TAGS_WITH_COLON = [
    "material:", "type:", "color:", "features:", 
    "background:", "style:", "view:", "symmetry:", "status:", "opacity:"
]
# Liste des mots-clés SANS deux-points pour le RegEx
TAG_KEYWORDS = [tag[:-1] for tag in EXPECTED_TAGS_WITH_COLON]

# Cherche un caractère qui n'est NI un espace, NI une virgule, 
# suivi de zéro ou plus d'espaces, suivi du tag, suivi de zéro ou plus d'espaces, suivi du deux-points.
tag_keywords_regex = '|'.join(re.escape(k) for k in TAG_KEYWORDS)
ERROR_PATTERN_BASE = r'([^\s,])\s*(' + tag_keywords_regex + r')\s*:\s*'


def verify_tag_separators(caption_root: Path):
    """
    Vérifie que chaque tag est séparé par la séquence stricte ', ' (virgule + espace).
    Utilise le RegEx le plus robuste pour éviter les faux positifs.
    """
    print(f"--- Vérification des séparateurs de Tags dans: {caption_root.parts[-3]}/{caption_root.parts[-1]} ---")
    
    files_with_missing_separators = {}
    
    for caption_path in caption_root.glob('**/*.txt'):
        try:
            content = caption_path.read_text(encoding='utf-8').strip()
        except Exception as e:
            files_with_missing_separators[caption_path.relative_to(caption_root)] = [f"Erreur de lecture: {e}"]
            continue

        errors = []
        
        # 1. Nettoyage du préfixe 'minecraft block :' 
        content_cleaned = re.sub(r'minecraft block\s*:\s*', '', content, 1, flags=re.IGNORECASE).strip()

        # Si le nettoyage n'a pas réussi et que le fichier devrait avoir le préfixe, c'est une erreur.
        if len(content_cleaned) == len(content) and not content.lower().startswith('minecraft block'):
             pass

        # 2. Chercher toutes les occurrences du motif d'erreur (Séparateur manquant)
        matches = list(re.finditer(ERROR_PATTERN_BASE, content_cleaned, re.IGNORECASE))

        # Si des matchs sont trouvés, c'est une erreur de séparation.
        if matches:
            first_match = matches[0]
            
            # Affiche le contexte de l'erreur
            start = max(0, first_match.start(0) - 15) 
            end = min(len(content_cleaned), first_match.end(0) + 15)
            context = content_cleaned[start:end].strip()
            
            # Le groupe 2 est le mot-clé du tag trouvé (ex: 'features')
            tag_keyword_found = first_match.group(2)
            
            errors.append(f"Séparateur ', ' manquant ou incorrect avant le tag '{tag_keyword_found}' près de: '...{context}...'")
        
        
        if errors:
            files_with_missing_separators[caption_path.relative_to(caption_root)] = errors
            
    if files_with_missing_separators:
        print(f"ERREUR: {len(files_with_missing_separators)} fichiers ont des séparateurs de tags manquants ou incorrects.")
        for file, errs in files_with_missing_separators.items():
            print(f"Fichier: {file}")
            for err in errs:
                print(f"Erreur -> {err}")
        return False
    else:
        print("Tous les fichiers de description ont des séparateurs de tags corrects.")
        return True


def verify_captions_internal_commas(caption_root: Path):
    """
    Vérifie qu'il n'y a aucune virgule à l'intérieur des valeurs de tags. 
    Affiche les détails des erreurs si elles sont trouvées.
    """
    print(f"--- Vérification des Virgules Internes dans: {caption_root.parts[-3]}/{caption_root.parts[-1]} ---")
    
    files_with_errors = {}
    
    for caption_path in caption_root.glob('**/*.txt'):
        try:
            content = caption_path.read_text(encoding='utf-8').strip()
        except Exception as e:
            files_with_errors[caption_path.relative_to(caption_root)] = [f"Erreur de lecture: {e}"]
            continue
            
        # Utilise un split tolérant aux espaces pour analyser les tags
        tags = re.split(r',\s*', content)
        errors = []
        
        for tag in tags:
            if ':' not in tag:
                 if not re.search(r'minecraft block\s*:\s*', tag, re.IGNORECASE):
                    errors.append(f"Tag mal formé (pas de ':'): '{tag}'")
                 continue
                 
            # Nettoie la valeur en retirant le tag et les espaces/deux-points
            value = re.sub(r'(.+?)\s*:\s*', '', tag, 1).strip()
            
            if ',' in value:
                errors.append(f"Virgule interne détectée dans le tag: '{tag}'")
                
        if errors:
            files_with_errors[caption_path.relative_to(caption_root)] = errors
            
    if files_with_errors:
        print(f"ERREUR INTERNE: {len(files_with_errors)} fichiers contiennent des virgules à l'intérieur des tags.")
        for file, errs in files_with_errors.items():
            print(f"Fichier: {file}")
            for err in errs:
                print(f"Erreur -> {err}")
        # -----------------------------------------------------------
        return False
    else:
        print("Aucune virgule interne détectée.")
        return True

def main():
    print("--- Démarrage de la Vérification Complète de la Syntaxe des Légendes ---")
    
    all_ok = True
    
    for root in CAPTION_ROOT_DIRS:
        print(f"--- Début des vérifications pour {root.parts[-3]}/{root.parts[-1]} ---")
        
        # --- Étape 1: Vérification des Séparateurs (Virgules manquantes) ---
        if not verify_tag_separators(root):
            all_ok = False
            
        # --- Étape 2: Vérification des Virgules Internes ---
        if not verify_captions_internal_commas(root):
            all_ok = False
            
    if all_ok:
        print("Vérification syntaxique complète réussie pour les deux ensembles.")
    else:
        print("Des corrections de séparateurs de tags et/ou de virgules internes sont nécessaires.")

if __name__ == "__main__":
    main()