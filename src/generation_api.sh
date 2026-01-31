#!/bin/bash

# Lancement de l'API avec Uvicorn
uvicorn src.main.main:app --host 0.0.0.0 --port 8000 --workers 1