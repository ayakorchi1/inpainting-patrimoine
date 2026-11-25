"""Télécharge le dataset Moroccan Zellige depuis Kaggle dans data/raw/.

Prérequis :
  1. pip install kaggle
  2. Sur kaggle.com > Settings > API > « Create New Token » : cela télécharge kaggle.json.
  3. Placer kaggle.json dans  C:\\Users\\<vous>\\.kaggle\\kaggle.json  (Windows)
     ou ~/.kaggle/kaggle.json (Linux/Mac). Ne JAMAIS le mettre dans Git.

Utilisation (depuis la racine du projet) :
  python data/download.py
"""
from pathlib import Path

DATASET = "anassiguedmi/moroccan-zellige"
DEST = Path(__file__).resolve().parent / "raw"


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()                      # lit kaggle.json
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Téléchargement de {DATASET} vers {DEST} ...")
    api.dataset_download_files(DATASET, path=str(DEST), unzip=True, quiet=False)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in DEST.rglob("*") if p.suffix.lower() in exts]
    n_true = sum(1 for p in images if "true" in p.parts)
    print(f"Terminé : {len(images)} images, dont {n_true} 'true'.")


if __name__ == "__main__":
    main()
