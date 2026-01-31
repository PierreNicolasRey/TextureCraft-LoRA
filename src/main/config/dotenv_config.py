import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

def get_env_path(key):
    val = os.getenv(key)
    if not val:
        print(f"{key} non trouvé dans le .env")
        return Path(".")
    return Path(val)