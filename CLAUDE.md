# Restaurer le patrimoine par inpainting

Projet étudiant, réalisé **seule** par l'étudiante. Elle débute : expliquer chaque action en français, simplement.
Environnement local : Windows + PowerShell.

## Règle
- **Après chaque étape, mettre à jour `docs/explications.md`** (carnet de préparation à la soutenance : ce qu'on a fait,
  pourquoi, notions clés sans jargon, résultats et interprétation, 3 questions probables du jury avec réponses).

## Objectif
À partir d'une image de zellige abîmée, proposer **plusieurs complétions cohérentes** de la zone manquante.

## Exigences du guide
- Comparer 3 méthodes : **OpenCV simple** (Telea / NS), **GAN**, **VQ-VAE + PixelCNN**.
- Étudier l'effet de la **taille du trou**.
- Évaluation chiffrée : **PSNR, SSIM, LPIPS, diversité, temps, mémoire GPU** + exemples visuels.
- Tout doit tourner sur **Kaggle/Colab avec 12 Go de VRAM max**.
- Démo **Gradio** : charger une image, marquer une zone, obtenir **4 propositions**.
- Rapport de 6-8 pages ; soutenance de 10 min.
- `docs/` : licences, fiche du modèle (model card), réflexion éthique.
- Dépôt organisé : `README.md`, `notebooks/`, `src/`, `configs/`, `data/`, `results/`, `demo/`, `report/`, `docs/`.
- **Pas de données ni de modèles lourds dans Git** (`data/raw/`, `data/processed/`, `*.pt`, `*.pth`, `*.ckpt` sont ignorés).
- Commits réguliers avec des messages précis.

## Données
- Dataset Kaggle `anassiguedmi/moroccan-zellige` : 1 560 images 256×256.
- On garde seulement les **983 images `true`** (vrais zelliges).
- Découpage d'origine conservé : **697 train / 143 validation / 143 test**.
- Redimensionnement en **128×128** (bicubique, RGB).
- Masques de test fixes (graine **2026**) : `central`, `carre`, `irregulier` × **10 / 20 / 40 %** → 9 jeux.
- Convention des masques : `uint8`, **1 = pixel effacé**, 0 = pixel connu.
- Détails : `data/README.md`.

## Organisation du code
- Les entraînements tournent sur **Kaggle (GPU)**.
- Le code réutilisable est dans `src/` ; les notebooks l'importent (ne pas dupliquer le code dans les notebooks).
  - `src/masks.py` : `masque_central`, `masque_carre`, `masque_irregulier`, `creer_masque`, `masque_aleatoire` (entraînement : type au hasard, taille 5-45 %).
  - `src/metrics.py` : `psnr_trou` (PSNR sur les pixels du trou uniquement), `ssim_img`.
- `data/download.py` : téléchargement via l'API Kaggle.

## Déjà fait
- `notebooks/01_exploration_preparation.ipynb` : exploration, sélection des 983 `true`, redimensionnement 128×128, masques de test.
- `notebooks/02_baseline_opencv.ipynb` : baseline OpenCV Telea/NS. PSNR dans le trou ≈ 12 dB ; SSIM 0,91 → 0,61 de 10 % à 40 % de trou. Résultats dans `results/`.

## À faire
- GAN, VQ-VAE + PixelCNN, LPIPS / diversité / mémoire GPU, démo Gradio, docs/, rapport.
