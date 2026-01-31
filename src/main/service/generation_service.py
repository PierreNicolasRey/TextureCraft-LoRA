import torch
import os
from io import BytesIO
import base64
from diffusers import StableDiffusionPipeline
from fastapi import HTTPException
from PIL import Image
import re
from typing import Optional, List, Dict, Any
from src.main.model.schemas import SuccessResponse
from src.main.config.settings import LORA_ROOT_DIR, BASE_MODEL_ID, DEVICE, DTYPE, MODEL_VERSIONS, DEFAULT_NEG_PROMPT, DEFAULT_GUIDANCE_SCALE, DEFAULT_STEPS, DEFAULT_SEED, CHROMA_KEY_COLOR, RGB_TOLERANCE, COLOR_TARGETS

pipe: StableDiffusionPipeline | None = None
lora_actif: str | None = None

def load_base_model():
    """Initialise le pipeline Stable Diffusion (Modèle de base)."""
    global pipe
    if pipe is not None:
        return

    print("-> Chargement du pipeline Stable Diffusion (Base)...")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=DTYPE,
            safety_checker=None
        )
        pipe.to(DEVICE)
        print("-> Pipeline de base prêt.")
    except Exception as e:
        print(f"Erreur fatale au chargement du modèle de base : {e}")
        # Relancer l'exception pour arrêter le serveur si le modèle de base ne peut pas être chargé
        raise RuntimeError("Impossible de charger le modèle de base SD.") from e


def generer_texture(prompt: str, model_version: str, resolution_cible: str, negative_prompt: str = DEFAULT_NEG_PROMPT, seed: int = DEFAULT_SEED) -> SuccessResponse:
    """
    Génère une image, en gérant le LoRA Switch.
    Retourne l'image encodée en Base64.
    """
    global pipe, lora_actif

    if pipe is None:
        raise HTTPException(status_code=500, detail="Le modèle de base n'est pas chargé.")

    # 1. Obtenir le chemin du LoRA
    if model_version not in MODEL_VERSIONS:
        raise HTTPException(status_code=400, detail="Version du modèle invalide.")
    
    lora_filename = MODEL_VERSIONS[model_version]
    lora_path = os.path.join(LORA_ROOT_DIR, lora_filename)

    if not os.path.exists(lora_path):
        raise HTTPException(status_code=500, detail=f"Fichier LoRA introuvable: {lora_path}")

    # 2. LO-RA SWITCH : Chargement CONDITIONNEL
    if model_version != lora_actif:
        # Seul le changement de modèle déclenche le chargement long
        
        lora_filename = MODEL_VERSIONS[model_version]
        lora_path = os.path.join(LORA_ROOT_DIR, lora_filename)

        if not os.path.exists(lora_path):
            raise HTTPException(status_code=500, detail=f"Fichier LoRA introuvable: {lora_path}")

        print(f"-> Changement de modèle LoRA : Chargement de '{model_version}'...")
        try:
            pipe.load_lora_weights(lora_path)
            
            # Mise à jour de l'état après succès
            lora_actif = model_version
            
            print(f"-> Modèle '{model_version}' chargé et actif.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur lors du chargement des poids LoRA: {e}")
    else:
        # Le LoRA est déjà celui demandé, on saute l'opération 'coûteuse'
        print(f"-> Modèle LoRA '{model_version}' déjà actif. Génération immédiate.")

    # 3. Préparation pour la génération
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    
    # 4. Génération
    print("Lancement de la génération de l'image.")
    try:
        image_512 = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=DEFAULT_STEPS,
            guidance_scale=DEFAULT_GUIDANCE_SCALE,
            generator=generator
        ).images[0]

        # 5. Transformation RGB -> RGBA et downscale
        image_512 = apply_chroma_key(image_512, prompt)

        image_cible = redimensionner_texture(image_512, resolution_cible)

        # 6. Conversion en Base64
        print("Transformation des images en base64.")
        buffered_cible = BytesIO()
        image_cible.save(buffered_cible, format="PNG")
        image_cible_str = base64.b64encode(buffered_cible.getvalue()).decode()

        buffered_affichage = BytesIO()
        image_512.save(buffered_affichage, format="PNG")
        image_512_str = base64.b64encode(buffered_affichage.getvalue()).decode()

        return SuccessResponse(
            imageBase64Cible= image_cible_str,
            imageBase64Affichage= image_512_str,
            model= model_version
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération: {e}")

def redimensionner_texture(image_512: Image, resolution_cible: str) -> Image:
    """
    Redimensionne l'image PIL 512x512 vers la résolution cible (ex: '64x64').
    Utilise l'algorithme NEAREST pour maintenir le style pixelisé.
    """
    print("Redimensionnement de l'image.")
    if not isinstance(image_512, Image.Image):
        raise TypeError("L'input n'est pas un objet PIL Image.")

    if not re.match(r'^\d+x\d+$', resolution_cible.lower()):
        raise HTTPException(status_code=400, detail="Format de résolution invalide ou valeurs non positives. Attendu : 'WxH' (W, H > 0).")

    width, height = map(int, resolution_cible.lower().split('x'))
    
    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail="La résolution cible doit être positive et non nulle.")

    resized_image = image_512.resize(
        (width, height), 
        resample=Image.Resampling.NEAREST
    )
    
    return resized_image

def apply_chroma_key(image_rgb: Image, prompt: str) -> Image:
    """ Applique la stratégie de post-traitement basée sur les tags du prompt. """
    
    # Extraction Sémantique du Prompt
    print("Récupération de la couche alpha de l'image.")

    infos_prompt = extract_prompt_infos(prompt)
    
    image_rgba = image_rgb.convert("RGBA")
    data = image_rgba.getdata()
    new_data = []
    
    r_key, g_key, b_key = CHROMA_KEY_COLOR

    colors_list = infos_prompt.get('colors')

    if (infos_prompt.get('is_opacity_transparent') or infos_prompt.get('is_opacity_semi_transparent')) and colors_list:
        color_target_str = colors_list[0]
        if color_target_str in COLOR_TARGETS:
            r_tex, g_tex, b_tex = COLOR_TARGETS[color_target_str]
        else:
            # Si la première couleur n'existe pas, on prend une couleur de secours ou on passe en mode binaire
            color_target_str = None
        
        for item in data:
            r, g, b, a_old = item
            
            # Calcul de l'Alpha basé sur la composante ayant la plus grande différence
            # On prend la composante où P_texture != C_key pour éviter la division par zéro.
            
            alpha_r = (r - r_key) / (r_tex - r_key) if (r_tex - r_key) != 0 else 1.0
            alpha_g = (g - g_key) / (g_tex - g_key) if (g_tex - g_key) != 0 else 1.0
            alpha_b = (b - b_key) / (b_tex - b_key) if (b_tex - b_key) != 0 else 1.0

            # Trouver le meilleur Alpha (le plus représentatif de l'opacité)
            alphas = [a for a in [alpha_r, alpha_g, alpha_b] if 0.0 <= a <= 1.0]
            
            # Si aucune alpha n'est valide (cas de bruit), on prend opaque
            alpha_new_float = min(alphas) if alphas else 1.0 

            alpha_brut_int = max(0, min(255, int(alpha_new_float * 255)))
            
            # Correction de Couleur (P_texture = (P_generated - (1-alpha)*C_key) / alpha)
            # Puisque nous avons l'Alpha, nous calculons la couleur P_texture
            alpha_norm = alpha_brut_int / 255.0
            
            if alpha_norm > 0:
                closest_color_name = find_closest_color_target(r, g, b, colors_list)
                
                if closest_color_name:
                    r_new, g_new, b_new = COLOR_TARGETS[closest_color_name] 
                else:
                    # Cas d'erreur : couleur non trouvée, on garde la couleur générée
                    r_new, g_new, b_new = r, g, b
    
            else:
                # Si l'alpha est 0 (totalement transparent), la couleur n'a pas d'importance
                r_new, g_new, b_new = r, g, b

            # APPEND : La couleur corrigée et l'Alpha final modulé
            new_data.append((r_new, g_new, b_new, alpha_brut_int))

    else:
        for item in data:
            r, g, b, a_old = item 
            
            r_ok = abs(r - r_key) <= RGB_TOLERANCE
            g_ok = abs(g - g_key) <= RGB_TOLERANCE
            b_ok = abs(b - b_key) <= RGB_TOLERANCE

            if r_ok and g_ok and b_ok:
                new_data.append((r, g, b, 0))
            else:
                new_data.append((r, g, b, 255)) 

    image_rgba.putdata(new_data)
    return image_rgba

def extract_prompt_infos(prompt: str) -> Dict[str, Any]:
    """
    Analyse le prompt pour extraire les informations de structure, d'opacité et de couleurs.
    """
    prompt_lower = prompt.lower()
    
    # --- 1. Extraction Binaire des Tags (Flags) ---
    has_background_tag = re.search(r'background\s*:\s*transparent', prompt_lower) is not None
    is_opacity_transparent = re.search(r'opacity\s*:\s*transparent', prompt_lower) is not None
    is_opacity_semi_transparent = re.search(r'opacity\s*:\s*semi-transparent', prompt_lower) is not None
    
    # --- 2. Extraction du Tag Couleurs (Liste) ---
    color_tag_match = re.search(r'color\s*:\s*([\w\s]+?)(?:,|\Z)', prompt_lower)
    
    colors: Optional[List[str]] = None
    if color_tag_match:
        # Récupère le contenu, enlève les espaces de début/fin, et divise par les espaces
        colors = color_tag_match.group(1).strip().split() 
        
    return {
        "has_background_tag": has_background_tag,
        "is_opacity_transparent": is_opacity_transparent,
        "is_opacity_semi_transparent": is_opacity_semi_transparent,
        "colors": colors
    }

def find_closest_color_target(r, g, b, color_names: List[str]) -> str:
    """
    Trouve la couleur cible (dans COLOR_TARGETS) qui est la plus proche
    du pixel généré (r, g, b) parmi les noms de couleurs du prompt.
    Retourne le nom de la couleur la plus proche.
    """
    if not color_names:
        return None

    min_distance = float('inf')
    closest_color_name = None
    
    for name in color_names:
        if name in COLOR_TARGETS:
            r_target, g_target, b_target = COLOR_TARGETS[name]
            
            # Calcul de la distance euclidienne dans l'espace RGB
            # distance² = (r - r_target)² + (g - g_target)² + (b - b_target)²
            distance_sq = (r - r_target)**2 + (g - g_target)**2 + (b - b_target)**2
            
            if distance_sq < min_distance:
                min_distance = distance_sq
                closest_color_name = name
                
    return closest_color_name