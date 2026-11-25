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
  - `src/metrics.py` : `psnr_trou` (PSNR sur les pixels du trou uniquement), `ssim_img`, `diversite_trou` (écart moyen entre propositions dans le trou, échelle 0-255).
  - `src/data.py` : `ZelligeDataset` (npz → [-1, 1], augmentation flips/rot90 sur train seulement, masque aléatoire neuf à chaque appel ou masques fixes), `trouver_npz`, `vers_tenseur`, `vers_uint8`.
  - `src/models/gan.py` : `Generateur` (U-Net compact, entrée 5 canaux = image trouée + masque + bruit, sortie tanh composée avec les pixels connus) et `Discriminateur` (PatchGAN 14×14, normalisation spectrale, entrée image + masque).
  - `src/train_gan.py` : `entrainer(config, npz)` (L1 trou × 100 + hinge ; éval PSNR/SSIM val toutes les 10 epochs, masque central 20 % fixe), `completer`, `generer`.
- `configs/` : un fichier YAML d'hyperparamètres par méthode (`gan.yaml`).
- Sorties d'entraînement (`.pt`, CSV, JSON mémoire GPU) écrites dans `/kaggle/working`, **jamais dans Git**. Seuls les CSV/PNG légers sont recopiés à la main dans `results/`.
- Les notebooks Kaggle commencent par `git clone` du dépôt + `sys.path`, puis `trouver_npz` pour localiser les données dans `/kaggle/input`.
- `data/download.py` : téléchargement via l'API Kaggle.
- Tests locaux : PyTorch n'est pas installé sur le PC ; tester sur CPU avec de fausses données (une itération), sans lancer l'entraînement complet.

## Déjà fait
- `notebooks/01_exploration_preparation.ipynb` : exploration, sélection des 983 `true`, redimensionnement 128×128, masques de test.
- `notebooks/02_baseline_opencv.ipynb` : baseline OpenCV Telea/NS. PSNR dans le trou ≈ 12 dB ; SSIM 0,91 → 0,61 de 10 % à 40 % de trou. Résultats dans `results/`.
- `notebooks/03_gan.ipynb` : **GAN v1 entraîné** (150 epochs, 18 min, pic GPU 2,03 Go, meilleur modèle epoch 110). PSNR +0,47 dB en moyenne vs Telea (9/9 configs), SSIM légèrement meilleur, 1,9 ms/image. Diversité faible (7-10/255) : propositions presque identiques. Visuel : bonne texture et bonnes couleurs, mais quadrillage, pas d'étoiles, raccord visible. Fichiers : `results/*_v1.*`.
- Convention : les résultats d'une version de modèle sont suffixés `_vN` dans `results/` (ex. `resultats_gan_v1.csv`, `memoire_gpu_gan_v1.json`). Les fichiers enregistrés depuis le navigateur peuvent arriver avec une extension `.txt` : vérifier l'extension et le contenu.

## À faire
- VQ-VAE + PixelCNN ; LPIPS ; éventuel GAN v2 (entraînement plus long, sélection sur SSIM/LPIPS, diversité) ; démo Gradio ; docs/ (fiche modèle, éthique) ; rapport.
