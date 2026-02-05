# TextureCraft-LoRA
> Python API application for Minecraft-like texture generation

---

## Description
TextureCraft-LoRA est une API Python conçue pour générer des textures de type "Minecraft" en utilisant des modèles LoRA entraînés spécifiquement. Cette application sert de moteur de génération pour un back-end Java.

### Liens du projet :
Front : https://github.com/PierreNicolasRey/TextureCraft-Front

Back : https://github.com/PierreNicolasRey/TextureCraft-Back
---

## Architecture du projet
- src/ : Contient l'intégralité du code source de l'API.

- utils/ : Scripts utilitaires pour le traitement, la vérification et le formatage des datasets. Ces scripts sont hors flux API et servent à la préparation des données.

- datasets/ : Contient les données (images et captions) d'entraînement et de contrôle (résolutions 16x16 et 512x512).

- latest_training_images/ : Échantillons d'images issus des trois derniers cycles d'entraînement pour suivi de progression.

- training/ : Contient les scripts d'entrainement :
    - TextureCraft_LoRA.ipynb : Notebook utilisé pour l'entrainement des modèles via Google Colab
    - train_dreambooth_lora.py : Script d'entrainement modifié pour répondre à mes besoins (récupère les images / captions récursivement dans leur dossier concept plutôt que depuis un seul répertoire)

---

## Dépendances
Le projet nécessite Python 3.x et les bibliothèques listées dans le fichier requirements.txt.

---

## Installation et Utilisation
- Installer les dépendances : pip install -r requirements.txt.

- Lancer l'API via le script shell : ./run_api.sh.

Si nécessaire, pour tester les fichiers utilitaires :
- Créer un fichier .env à la racine en vous basant sur .env.example pour configurer vos chemins locaux.

---

## Structure des Captions
Contrairement à un entraînement classique en langage naturel, ce modèle utilise un système de tags hiérarchisés. Chaque image est décrite par un ensemble de mots-clés (tags) séparés par des virgules, classés du plus général au plus spécifique.
J'utilise des tags obligatoires et des tags optionnels.

- Tags obligatoires :
    - **minecraft block:**
    - **material:**
    - **type:**
    - **color:**
    - **features:**
    - **style: pixel art**

- Tags optionnels :
    - **background: transparent**
    - **opacity:** Valeur : transparent ou semi-transparent
    - **view:** L'orientation de la vue (top, bottom, north, etc.)
    - **symmetry:**


Exemple de prompt : "minecraft block: material: terracotta, type: orange terracotta, color: reddish orange, features: solid reddish orange surface with subtle pixel variation, style: pixel art"
Ce prompt sert aussi de test pour produire les images des modèles nouvellement entrainés disponibles dans latest_training_images/.

Ce format permet un contrôle plus granulaire lors de la génération via l'API, en facilitant l'interpolation entre différents concepts de textures.

---

## Pour aller plus loin
A ce jour, les entrainements ne sont pas terminés. Les images simples en sortie ne correspondent pas à l'attendu. Tant que le stade de la correspondance parfaite entre une image **existante** et une image **générée** selon la description de cette première n'est pas atteint, je ne pourrais pas considérer que mon modèle connaisse parfaitement le style Minecraft que je souhaite lui apprendre.

Il est donc nécessaire d'atteindre ce stade avant de ne pouvoir demander à l'application des images totalement inventées et pourtant dans le style voulu.

De plus, les modèles issus de mes entrainements ne sont entrainés que sur les blocs de Minecraft et pas les items. Un modèle existe déjà pour ceux-ci. Une fois mon propre modèle affermit, je pourrais réfléchir à une intégration de ce second modèle dans l'application avec l'accord de son auteur.

Enfin, j'aimerais proposer différents modèles de génération d'image de bloc. Celui entrainé actuellement est un modèle "Vanilla", entrainé uniquement sur les images de blocs du jeu de base. Des sur-couches LoRA sur ce modèle Vanilla sont à envisager pour différents mods et ainsi permettre la croissance d'add-ons qui aurait pu se trouver limitée par le manque de textures à appliquer. Je pense ainsi au mod Create à l'esthétique particulière et à ses nombreux add-ons qui pourrait bénéficier d'un modèle spécialement conçu pour cette esthétique.
