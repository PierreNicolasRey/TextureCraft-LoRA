import os
from pathlib import Path
from dotenv import load_dotenv

def find_dotenv():
    current = Path(__file__).resolve().parent
    for _ in range(5): # Cherche jusqu'à 5 niveaux au-dessus
        if (current / '.env').exists():
            return current / '.env'
        current = current.parent
    return None

env_path = find_dotenv()
load_dotenv(dotenv_path=env_path)

def get_env_path(key):
    val = os.getenv(key)
    if not val:
        print(f"{key} non trouvé dans le .env")
        return Path(".")
    return Path(val)