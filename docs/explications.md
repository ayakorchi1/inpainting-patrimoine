# Carnet d'explications — préparation de la soutenance

Projet « Restaurer le patrimoine par inpainting », réalisé seule.
Ce carnet est mis à jour après chaque étape. Pour chaque étape, il présente :
ce qui a été fait, pourquoi, les notions clés, les résultats et 3 questions probables du jury.

**Rappel de l'objectif :** à partir d'une photo de zellige abîmée, proposer **plusieurs** complétions
cohérentes de la zone manquante. On compare trois méthodes : OpenCV (simple), GAN, VQ-VAE + PixelCNN.

---

## 1) Données

### Ce qu'on a fait
- On a utilisé le dataset Kaggle **Moroccan Zellige** (`anassiguedmi/moroccan-zellige`) : 1 560 images de 256×256 pixels.
- On a gardé seulement les **983 images « true »**, c'est-à-dire les vrais zelliges.
- On a **redimensionné** les images en **128×128** pixels.
- On a gardé le **découpage d'origine** : **697** images pour l'entraînement, **143** pour la validation et **143** pour le test.
- Le tout est enregistré dans un fichier `zellige_128.npz` (notebook 01).

### Pourquoi
- **Seulement les « true » :** le dataset contient aussi des images « false », qui ne sont pas des zelliges
  (autres motifs, objets, imitations). Notre but est de restaurer des zelliges. Si le modèle apprenait aussi
  sur des images qui n'en sont pas, il risquerait de compléter un trou avec un motif qui n'a rien à voir.
- **128×128 :** on doit tout faire tourner sur Kaggle/Colab avec au maximum 12 Go de mémoire graphique.
  Une image de 128×128 contient 4 fois moins de pixels qu'une image de 256×256 : l'entraînement est plus rapide
  et consomme moins de mémoire. C'est un compromis : on perd un peu de détail fin.
- **Découpage d'origine :** il est fourni avec le dataset, donc reproductible. Surtout, il garantit que les images
  de test ne sont **jamais vues** pendant l'entraînement.

### Notions clés
- **Entraînement / validation / test :**
  - *Entraînement :* les images sur lesquelles le modèle apprend.
  - *Validation :* les images qui servent à régler le modèle pendant le travail (choisir les réglages, savoir quand arrêter).
  - *Test :* les images gardées « sous clé », utilisées une seule fois à la fin pour la note finale. C'est comme un examen
    dont on ne connaît pas les questions à l'avance.
- **Sur-apprentissage :** le modèle apprend « par cœur » les images d'entraînement et se trompe sur les nouvelles.
  Avec seulement 697 images, c'est un vrai risque.
- **Licence ODbL / DbCL :** deux licences libres d'Open Data Commons. L'ODbL couvre la base de données
  (l'ensemble organisé). La DbCL couvre le contenu (les images). Elles autorisent l'usage si on cite la source.

### Licence et réserve
- Le dataset est sous **ODbL 1.0** pour la base et **DbCL 1.0** pour le contenu.
- **Réserve :** les images semblent venir du web, car on voit des filigranes. La personne qui a publié le dataset
  n'a probablement pas pris toutes ces photos elle-même. On ne peut donc pas garantir les droits sur chaque image.
- **Ce que j'ai fait :** je ne redistribue pas les images dans mon dépôt Git. Chacun les télécharge depuis Kaggle,
  et je cite la source. Les détails sont dans `docs/licences.md`.

### Résultats / constats
- 983 images utilisables (697 / 143 / 143), toutes converties en couleur RGB, en 128×128.
- Limites observées : **filigranes** sur certaines images, petit volume de données, détails fins perdus en 128×128.

### Questions probables du jury
1. **« Pourquoi ne pas avoir gardé toutes les images, y compris les "false" ? »**
   Parce qu'elles ne sont pas des zelliges. Le modèle doit apprendre les règles propres aux zelliges :
   symétries, motifs géométriques, palette de couleurs. Mélanger d'autres images brouillerait cet apprentissage
   et pourrait produire des complétions incohérentes.
2. **« 128×128, ce n'est pas trop petit pour du patrimoine ? »**
   Si, c'est une limite assumée. Elle est imposée par la contrainte de 12 Go de mémoire graphique et par le temps
   d'entraînement disponible sur Kaggle. Le but du projet est de **comparer** des méthodes dans les mêmes conditions,
   pas de produire une restauration en haute définition. Monter en résolution est une piste d'amélioration.
3. **« Avez-vous le droit d'utiliser ces images ? »**
   Le dataset est publié sous licences ODbL/DbCL, qui autorisent l'usage avec citation. J'ai cependant noté une réserve :
   les filigranes montrent que certaines photos viennent du web, donc leurs droits individuels ne sont pas garantis.
   C'est pourquoi je ne redistribue pas les images. Je les utilise dans un cadre pédagogique, en citant la source.

---

## 2) Masques

### Ce qu'on a fait
- Un **masque** est une image noir et blanc de la même taille que la photo : **1 = pixel effacé** (le trou), 0 = pixel connu.
- On a créé **3 types** de masques :
  - **central :** un carré au milieu de l'image ;
  - **carré :** un carré placé au hasard ;
  - **irrégulier :** des traits épais dessinés au hasard, qui imitent des fissures ou des éclats.
- On a utilisé **3 tailles** de trou : **10 %, 20 % et 40 %** de l'image.
- Cela donne 3 × 3 = **9 configurations de test**, appliquées aux 143 images de test.
- Les masques de test sont **fixes** : ils sont générés avec une graine aléatoire fixée (**2026**) et enregistrés
  dans `masques_test_128.npz`.
- Pour l'entraînement, on tire un masque **différent à chaque fois** : type au hasard, taille entre 5 et 45 %
  (fonction `masque_aleatoire` dans `src/masks.py`).

### Pourquoi
- **Des masques artificiels :** pour mesurer la qualité d'une restauration, il faut connaître la « bonne réponse ».
  Sur une vraie photo abîmée, on ne sait pas ce qu'il y avait sous la partie détruite. En effaçant nous-mêmes une zone
  d'une image intacte, on garde l'original pour comparer. Cela permet aussi de choisir exactement la forme et la taille du trou.
- **Trois types :** ils représentent des situations différentes. Le carré central est le cas le plus dur (loin des bords,
  peu de contexte). Le carré au hasard vérifie que la méthode marche partout. Le masque irrégulier ressemble davantage
  à de vrais dégâts.
- **Trois tailles :** le guide demande d'étudier l'**effet de la taille du trou**. Plus le trou est grand, plus il faut « inventer ».
- **Masques de test fixes :** toutes les méthodes (OpenCV, GAN, VQ-VAE) sont évaluées sur **exactement les mêmes trous**.
  C'est une comparaison équitable, comme donner le même examen à tous les élèves. Si on tirait de nouveaux masques à chaque fois,
  une différence de score pourrait venir du hasard des masques et non de la méthode.

### Notions clés
- **Graine aléatoire (seed) :** un nombre qui fixe le point de départ du générateur de hasard. Avec la même graine,
  on obtient toujours la même suite de nombres « au hasard ». Le résultat est donc **reproductible**.
- **Inpainting :** remplir la zone manquante d'une image de manière crédible.
- **Varier les masques à l'entraînement :** si le modèle ne voyait qu'un carré central de 20 %, il ne saurait rien faire d'autre.
  En variant forme et taille, on l'oblige à savoir gérer n'importe quel trou (y compris celui que l'utilisateur dessinera dans la démo).

### Résultats / constats
- Les surfaces obtenues sont bien proches de celles visées (≈ 10 %, 20 %, 40 %).
  Le masque irrégulier dépasse un peu, car on ajoute des traits jusqu'à atteindre au moins la surface voulue.

### Questions probables du jury
1. **« Vos masques artificiels sont-ils réalistes ? »**
   Pas totalement, et c'est une limite. Les vrais dégâts (usure, éclats, taches, mortier) ont des formes et des couleurs
   variées. Mais les masques artificiels sont indispensables pour **mesurer** la qualité, puisqu'on connaît l'original.
   Le masque irrégulier est un compromis qui se rapproche de fissures réelles.
2. **« Pourquoi fixer la graine des masques de test ? »**
   Pour que toutes les méthodes soient évaluées sur exactement les mêmes trous, et que n'importe qui puisse refaire
   l'expérience et retrouver mes chiffres. C'est la reproductibilité.
3. **« Pourquoi 10, 20 et 40 % ? »**
   Ce sont trois niveaux faciles à interpréter : petit, moyen et grand dégât. 40 % est déjà très difficile : presque la moitié
   de l'image est à inventer. Cela suffit pour voir comment chaque méthode se dégrade quand le trou grandit.
   À l'entraînement, on couvre toute la plage de 5 à 45 % pour que le modèle soit à l'aise entre ces valeurs.

---

## 3) Baseline OpenCV (Telea et Navier-Stokes)

### Ce qu'on a fait
- On a testé deux méthodes **classiques**, sans apprentissage, fournies par OpenCV : **Telea** et **Navier-Stokes (NS)**.
- On les a appliquées aux 143 images de test × 9 configurations de masques.
- On a mesuré le **PSNR dans le trou**, le **SSIM** sur l'image entière et le **temps** de calcul (notebook 02).
- Résultats dans `results/resultats_baseline.csv` ; figures `results/baseline_psnr_taille.png` et `results/baseline_exemples.png`.

### Pourquoi
- Une **baseline** (méthode de référence) sert de point de comparaison. Si un modèle d'apprentissage profond,
  long à entraîner, ne fait pas mieux qu'une méthode simple et instantanée, il n'a pas d'intérêt.
- Elle montre aussi concrètement **ce qui manque** aux méthodes simples, ce qui justifie la suite du projet.

### Notions clés
- **Principe de Telea :** l'algorithme remplit le trou **du bord vers le centre**, couche par couche, comme une tache d'encre
  qui s'étale à l'envers. Chaque pixel manquant reçoit une moyenne pondérée des pixels connus voisins, en suivant la direction
  des contours. Il **prolonge** ce qui est autour, mais **n'invente pas** de motif.
- **Navier-Stokes :** même idée, inspirée de l'écoulement des fluides ; il prolonge les lignes de contour dans le trou.
  En pratique, il donne ici des résultats quasi identiques à Telea.
- **PSNR (rapport signal sur bruit de crête),** en décibels (dB) : mesure l'écart pixel par pixel entre l'image restaurée et l'originale.
  **Plus c'est haut, mieux c'est.** Ici, on le calcule **uniquement sur les pixels du trou**, car c'est là que se trouve la restauration.
  Repères : moins de 20 dB, c'est très différent de l'original ; 30 dB et plus, c'est très proche.
- **SSIM (similarité structurelle),** entre 0 et 1 : compare la **structure** (contrastes, contours, textures) par petites
  fenêtres, plus proche de la perception humaine. 1 = identique. Il est calculé sur **l'image entière**.

### Résultats

PSNR moyen dans le trou (dB), méthode Telea :

| Type de masque | 10 % | 20 % | 40 % |
|---|---|---|---|
| central | 11,85 | 11,60 | 11,63 |
| carré | 12,01 | 11,80 | 11,57 |
| irrégulier | 12,47 | 12,16 | 11,98 |

SSIM moyen (image entière), méthode Telea :

| Type de masque | 10 % | 20 % | 40 % |
|---|---|---|---|
| central | 0,909 | 0,809 | 0,608 |
| carré | 0,910 | 0,812 | 0,609 |
| irrégulier | 0,912 | 0,830 | 0,656 |

- NS donne des chiffres quasi identiques (écart inférieur à 0,1 dB).
- Temps : environ **11 ms par image** pour chacune des deux méthodes, sur le processeur (CPU).

### Interprétation
- **PSNR ≈ 12 dB partout :** c'est **très bas**. Dans le trou, l'image restaurée est très différente de l'original.
  Visuellement, Telea produit une zone **floue et lissée**, sans les motifs géométriques du zellige.
- **Le PSNR varie peu avec la taille du trou.** Il est calculé seulement dans le trou, comme une note moyenne par pixel manquant.
  Même pour un petit trou, Telea ne retrouve pas le motif. La qualité par pixel est donc déjà mauvaise à 10 % et ne peut guère baisser.
- **Le SSIM chute de 0,91 à 0,61.** Il est calculé sur toute l'image. Plus le trou est grand, plus la part de l'image
  « ratée » est grande, donc plus le score global baisse. Le SSIM mesure à la fois la qualité et la **proportion** abîmée.
- **Les masques irréguliers sont un peu plus faciles :** chaque pixel manquant est proche d'un pixel connu (traits fins),
  alors qu'au centre d'un grand carré, l'information la plus proche est loin.

### Limites du PSNR et du SSIM
- **Ils récompensent le flou.** Une image floue qui ressemble « en moyenne » à l'original peut avoir un meilleur score
  qu'une complétion nette et crédible, mais avec un motif décalé de quelques pixels.
- **Une seule bonne réponse.** Ils comparent à l'original. Or notre objectif est de proposer **plusieurs** complétions
  cohérentes : une proposition différente de l'original mais tout à fait plausible sera pénalisée.
- **Ils ne mesurent pas la diversité** des propositions.
- C'est pourquoi on ajoutera **LPIPS** (une distance qui imite la perception humaine, grâce à un réseau de neurones),
  une mesure de **diversité**, et des **exemples visuels**.

### Pourquoi il faut des modèles appris
- Telea ne fait que **prolonger les bords** : il n'a aucune connaissance de ce qu'est un zellige
  (étoiles, symétries, répétitions, couleurs typiques).
- Un modèle **appris** (GAN, VQ-VAE + PixelCNN) a vu des centaines de zelliges. Il peut **inventer** un motif plausible
  au centre d'un grand trou.
- Telea est **déterministe** : il donne toujours la même réponse. Un modèle **génératif** peut proposer
  **plusieurs** complétions différentes, ce qui est précisément l'objectif du projet (4 propositions dans la démo).

### Questions probables du jury
1. **« 12 dB, c'est bon ou mauvais ? »**
   C'est mauvais : au-delà de 30 dB, on considère une image très fidèle. Mais il faut savoir que le PSNR est mesuré
   **uniquement dans le trou**, ce qui le rend plus sévère que sur l'image entière. Cela confirme ce qu'on voit à l'œil :
   Telea remplit avec un flou sans motif. C'est la référence que les modèles appris doivent battre.
2. **« Pourquoi le PSNR ne baisse presque pas quand le trou grandit, alors que le SSIM chute ? »**
   Parce qu'ils ne mesurent pas la même chose. Le PSNR est une moyenne **par pixel du trou** : Telea est déjà mauvais
   à 10 %, donc la note par pixel reste à peu près la même. Le SSIM est calculé sur **toute l'image** : quand le trou passe
   de 10 % à 40 %, la part ratée de l'image augmente, donc le score global baisse.
3. **« Si le PSNR et le SSIM ont des limites, pourquoi les utiliser ? »**
   Ce sont les mesures standard, faciles à calculer et comparables avec d'autres travaux. Je les complète avec LPIPS,
   plus proche de la perception humaine, avec une mesure de diversité (puisque je veux plusieurs propositions)
   et avec des exemples visuels. Aucune mesure ne suffit à elle seule.

---

## 4) GAN — *à compléter*

## 5) VQ-VAE + PixelCNN — *à compléter*

## 6) Comparaison finale (PSNR, SSIM, LPIPS, diversité, temps, mémoire GPU) — *à compléter*

## 7) Démo Gradio — *à compléter*

## 8) Éthique et limites — *à compléter*
