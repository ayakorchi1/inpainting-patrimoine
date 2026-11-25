# Fiche des données

Les images ne sont **pas** stockées dans Git (`data/raw/` et `data/processed/` sont ignorés).
Pour les récupérer : `python data/download.py` (voir plus bas).

## Source
- Dataset Kaggle : [anassiguedmi/moroccan-zellige](https://www.kaggle.com/datasets/anassiguedmi/moroccan-zellige)
- 1 560 images 256×256, réparties en dossiers `train / validation / test`, chacun séparé en `true` et `false`.
- **Licence :** ODbL 1.0 pour la base de données, DbCL 1.0 pour le contenu (Open Data Commons). Usage autorisé avec citation de la source. Réserve : les images semblent provenir du web (filigranes visibles) ; l'auteur du dataset n'est probablement pas l'auteur de chaque photo, les droits individuels ne sont donc pas garantis. Les images ne sont pas redistribuées dans ce dépôt. Détails : [docs/licences.md](../docs/licences.md).

## Sélection : 983 images `true`
- `true` = vrais zelliges ; `false` = autres images (motifs non zellige, objets, imitations…).
- On garde **uniquement les 983 images `true`** : le modèle doit apprendre la géométrie et les couleurs
  des vrais zelliges. Les `false` ajouteraient des motifs qui ne correspondent pas à ce qu'on veut restaurer.
- On conserve le **découpage d'origine** : 697 train / 143 validation / 143 test
  (pas de fuite entre entraînement et test, résultats comparables).

## Prétraitement (notebook 01)
- Conversion en RGB, redimensionnement en **128×128** (interpolation bicubique)
  pour tenir dans 12 Go de VRAM et entraîner rapidement.
- Sauvegarde : `zellige_128.npz` (clés `train`, `val`, `test`, tableaux `uint8` (N, 128, 128, 3)).
- Masques de test fixes (`masques_test_128.npz`), graine **2026** :
  3 types (`central`, `carre`, `irregulier`) × 3 tailles (10, 20, 40 %) = 9 clés, ex. `central_10`,
  chacune de forme (143, 128, 128). 1 = pixel effacé.
- Masques d'entraînement : tirés à la volée avec `src.masks.masque_aleatoire` (type au hasard, 5-45 %).

## Limites connues
- **Filigranes / logos** sur certaines images (images issues du web) : le modèle peut apprendre à les reproduire.
- Petite taille du dataset (697 images d'entraînement) → risque de sur-apprentissage.
- Résolution réduite à 128×128 : les détails fins des motifs sont perdus.
- Diversité limitée : photos de catalogue, angles et éclairages variables, peu d'images réellement abîmées.
  Les dégâts sont **simulés** par les masques, ce qui ne représente pas parfaitement l'usure réelle.
- Possibles doublons ou quasi-doublons entre images (non vérifié).
