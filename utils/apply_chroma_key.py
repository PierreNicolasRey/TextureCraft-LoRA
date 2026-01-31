from pathlib import Path
from PIL import Image
import re
import os
from typing import Optional, List, Dict, Any
from src.main.config.dotenv_config import get_env_path

# Ces valeurs sont les COULEURS ATTENDUES de la texture SANS la Chroma Key.
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
CHROMA_KEY_COLOR = (0, 255, 0)
RGB_TOLERANCE = 3

def main():
    file_path = Path(get_env_path("IMAGE_APPLY_CHROMA_KEY_TO"))
    base_image = Image.open(file_path)

    target_path = Path(get_env_path("IMAGE_RESULT_APPLY_CHROMA_KEY"))
    target_image = apply_chroma_key(base_image, "color: black, opacity: transparent")

    target_image.save(target_path, format='PNG')

def apply_chroma_key(image_rgb: Image, prompt: str) -> Image:
    """ Applique la stratégie de post-traitement basée sur les tags du prompt. """
    
    # --- 0. Extraction Sémantique du Prompt ---
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
            
            # 2. Correction de Couleur (P_texture = (P_generated - (1-alpha)*C_key) / alpha)
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

            # 4. APPEND : La couleur corrigée et l'Alpha final modulé
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
    has_background_tag = 'background: transparent' in prompt_lower
    is_opacity_transparent = 'opacity: transparent' in prompt_lower
    is_opacity_semi_transparent = 'opacity: semi-transparent' in prompt_lower
    
    # --- 2. Extraction du Tag Couleurs (Liste) ---
    color_tag_match = re.search(r'color:\s*([\w\s]+?)(?:,|\Z)', prompt_lower)
    
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
        return None # Ou lever une erreur

    min_distance = float('inf')
    closest_color_name = None
    
    pixel_generated = (r, g, b)

    for name in color_names:
        if name in COLOR_TARGETS:
            r_target, g_target, b_target = COLOR_TARGETS[name]
            
            # Calcul de la distance euclidienne dans l'espace RGB
            distance_sq = (r - r_target)**2 + (g - g_target)**2 + (b - b_target)**2
            
            if distance_sq < min_distance:
                min_distance = distance_sq
                closest_color_name = name
                
    return closest_color_name

if __name__ == "__main__":
    main()