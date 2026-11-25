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

## 4) GAN d'inpainting

> Statut : **GAN v1 entraîné sur Kaggle** (150 epochs, 18 min). Résultats dans `results/*_v1.*`.

### Ce qu'on a fait
- `src/data.py` : prépare les images pour PyTorch. Il passe les pixels de 0-255 à **-1 / +1**, applique des
  **retournements et rotations** (entraînement seulement) et associe à chaque image un **nouveau masque aléatoire** à chaque passage.
- `src/models/gan.py` : deux réseaux.
  - **Générateur** (U-Net) : il reçoit l'image trouée, le masque et une image de **bruit**, puis dessine le contenu du trou.
  - **Discriminateur** (PatchGAN) : il regarde une image et juge, zone par zone, si elle est « vraie » ou « fabriquée ».
- `configs/gan.yaml` : tous les réglages (taille 128, batch 32, 150 epochs, Adam lr 0,0002, poids L1 = 100, poids adverse = 1, graine 42).
- `src/train_gan.py` : la boucle d'entraînement. Toutes les 10 epochs, le script mesure PSNR/SSIM sur la validation avec un masque fixe,
  garde le meilleur modèle, enregistre l'historique des pertes et le **pic de mémoire GPU**.
- `notebooks/03_gan.ipynb` : entraînement sur Kaggle, courbes, évaluation sur les **mêmes 143 images × 9 masques** que la baseline,
  mesure de diversité (4 propositions par image) et figures.

### Principe du GAN (sans jargon)
Un GAN, c'est un **duel entre deux réseaux** :
- le **générateur** est un faussaire : il remplit le trou ;
- le **discriminateur** est un expert : il essaie de repérer les images retouchées.

À chaque tour, l'expert s'améliore pour démasquer le faussaire, et le faussaire s'améliore pour tromper l'expert.
À la fin, le générateur produit des remplissages assez réalistes pour que l'expert ne fasse plus la différence.

Si l'on ne demandait que « tromper l'expert », le faussaire pourrait inventer n'importe quel joli motif sans rapport avec l'image.
On ajoute donc une deuxième consigne, la **perte L1** : « ressemble à l'original dans le trou ».
Les deux consignes se complètent. La L1 apporte la fidélité (bonnes couleurs, bonne continuité), et le duel apporte la
netteté (des motifs francs au lieu du flou).

### Choix techniques et pourquoi
| Choix | Pourquoi |
|---|---|
| **U-Net** (générateur) | Il réduit l'image pour en comprendre la structure globale, puis la regrandit. Les **connexions de saut** recopient les détails fins des bords, ce qui assure un raccord propre avec les pixels connus. « Compact » : 5,3 millions de paramètres (et 2,8 millions pour le discriminateur), ce qui tient largement dans 12 Go. |
| **5 canaux en entrée** | Image trouée (3 couleurs) + **masque** (pour savoir *où* inventer) + **bruit** (pour pouvoir donner *plusieurs* réponses différentes). |
| **Sortie tanh, images en [-1, 1]** | tanh donne des valeurs entre -1 et 1 ; on met les images à la même échelle. |
| **Composition finale** | On garde les pixels connus tels quels et on ne prend le résultat du générateur **que dans le trou**. Le réseau ne peut pas abîmer ce qui était intact. |
| **PatchGAN** (discriminateur) | Au lieu d'une seule note pour toute l'image, il donne une note pour chaque zone (grille 14×14). Il juge ainsi la **texture locale** (les motifs du zellige), ce qui est exactement ce qu'on veut rendre réaliste. Il reçoit aussi le masque pour savoir où regarder. |
| **Normalisation spectrale** | Elle empêche le discriminateur de devenir trop sévère trop vite. Sinon, le générateur ne reçoit plus de signal utile et l'entraînement s'effondre. C'est un « frein » qui stabilise le duel. |
| **Perte hinge** | Une manière de noter le duel : on ne punit que les erreurs franches. Elle est stable et couramment associée à la normalisation spectrale. |
| **L1 × 100, adverse × 1** | La fidélité est prioritaire (réglage classique du modèle pix2pix) ; le duel ajoute la netteté. La L1 est calculée **dans le trou seulement**. |
| **Adam, lr 0,0002, betas 0,5 / 0,999** | Réglages standards pour les GAN ; un beta1 bas rend l'optimiseur plus réactif dans le duel. |
| **Augmentation** (retournements, rotations de 90°) | Les zelliges sont souvent symétriques : une rotation donne une nouvelle image tout aussi valable. On multiplie ainsi artificiellement les 697 images, ce qui limite le sur-apprentissage. Seulement sur l'entraînement, pour garder une évaluation honnête. |
| **Masque aléatoire neuf à chaque passage** | Le modèle voit des milliers de trous différents (forme et taille 5-45 %). Il apprend à gérer n'importe quel trou, y compris celui dessiné dans la démo. |
| **Évaluation avec masque et bruit fixes** | Pour que les mesures d'une epoch à l'autre soient comparables : seul le modèle change. |
| **Mémoire GPU mesurée** | Pour vérifier la contrainte du guide (12 Go max), avec `torch.cuda.max_memory_allocated`. |

### Notions clés
- **Epoch :** un passage complet sur les 697 images d'entraînement. 150 epochs = le modèle voit chaque image 150 fois
  (avec à chaque fois une rotation et un trou différents).
- **Batch :** le nombre d'images traitées ensemble (32) avant de corriger le réseau.
- **Perte (loss) :** un nombre qui dit « à quel point le réseau se trompe ». L'entraînement consiste à le faire baisser.
- **Effondrement de modes (mode collapse) :** risque connu des GAN. Le générateur apprend à **ignorer le bruit** et donne
  toujours la même réponse. C'est pour le détecter qu'on mesure la diversité (cellule 5) : un score proche de 0 signale ce problème.
- **Diversité :** on génère 4 propositions avec 4 bruits différents et on mesure de combien elles diffèrent entre elles dans le trou
  (en niveaux de 0 à 255).

### Résultats (GAN v1)
Fichiers : `results/resultats_gan_v1.csv`, `resultats_gan_diversite_v1.csv`, `historique_gan_v1.csv`,
`memoire_gpu_gan_v1.json`, `gan_courbes_v1.png`, `gan_exemples_v1.png`, `gan_propositions_v1.png`.
Même protocole que la baseline : 143 images de test × 9 masques fixes, bruit fixe (graine 0).

**PSNR dans le trou (dB) : GAN contre Telea**

| Masque | 10 % | 20 % | 40 % |
|---|---|---|---|
| central | 12,32 (Telea 11,85, **+0,47**) | 12,06 (11,60, **+0,47**) | 12,02 (11,63, **+0,39**) |
| carré | 12,48 (12,01, **+0,47**) | 12,24 (11,80, **+0,44**) | 12,07 (11,57, **+0,50**) |
| irrégulier | 12,85 (12,47, **+0,37**) | 12,71 (12,16, **+0,55**) | 12,59 (11,98, **+0,61**) |

→ Le GAN gagne sur **les 9 configurations**, en moyenne **+0,47 dB** (de +0,37 à +0,61 dB).

**SSIM (image entière) : GAN contre Telea**

| Masque | 10 % | 20 % | 40 % |
|---|---|---|---|
| central | 0,914 (0,909) | 0,817 (0,809) | 0,619 (0,608) |
| carré | 0,915 (0,910) | 0,819 (0,812) | 0,619 (0,609) |
| irrégulier | 0,917 (0,912) | 0,840 (0,830) | 0,673 (0,656) |

→ Légèrement meilleur partout (+0,005 à +0,017). Le gain grandit avec la taille du trou.

**Diversité des 4 propositions** (écart moyen dans le trou, échelle 0-255)

| Masque | 10 % | 20 % | 40 % |
|---|---|---|---|
| central | 9,7 | 9,5 | 8,0 |
| carré | 8,8 | 8,0 | 7,5 |
| irrégulier | 7,1 | 7,0 | 7,1 |

→ Entre **7 et 10 sur 255**, soit environ 3 à 4 % de la plage des couleurs. Pour comparer : l'erreur L1 finale
(0,385 sur l'échelle [-1, 1]) correspond à environ **49 sur 255**. Les propositions diffèrent donc environ
**5 fois moins entre elles qu'elles ne diffèrent de l'original**.

**Coût**
- Temps : **1,9 ms par image** sur GPU (contre environ 11 ms pour Telea sur CPU).
- Mémoire : pic de **2,03 Go**, bien sous la limite de 12 Go.
- Entraînement : **18 min** pour 150 epochs.

**Entraînement** (`gan_courbes_v1.png`)
- **Meilleur modèle à l'epoch 110** : PSNR de validation de 12,06 dB, SSIM de 0,819.
- La **L1 baisse encore à l'epoch 150** (de 0,487 à 0,385, sans plateau) : le modèle **n'a pas fini d'apprendre**.
- Le **PSNR de validation oscille** entre 11,66 et 12,06 dB, sans tendance nette, alors que le SSIM de validation
  monte régulièrement (de 0,804 à 0,823). L'écart de l'epoch 110 sur ses voisines (environ 0,2 dB) est donc
  en partie dû au hasard.
- Le duel reste équilibré : la perte du discriminateur se stabilise autour de 0,6-0,7 et ne tombe pas à 0.
  Le discriminateur ne « gagne » pas complètement.

### Analyse visuelle (`gan_exemples_v1.png`, `gan_propositions_v1.png`)
- ✅ **Texture et couleurs justes** : le GAN reprend la palette de l'image (beige, bleu-gris, brun). Telea, lui, étire
  les couleurs des bords en rayons flous.
- ❌ **Motif en quadrillage** : le trou est rempli d'un grain fin et régulier, une sorte de « bruit de zellige ».
  On ne voit **aucune étoile**, aucune ligne qui continue, aucune forme géométrique.
- ❌ **Mauvais raccord aux bords** : un cadre clair et flou marque nettement la limite du trou, surtout à 40 %.
  Les lignes qui arrivent au bord du trou ne se prolongent pas à l'intérieur.
- ❌ **4 propositions presque identiques** : même texture, mêmes couleurs, avec seulement de petites variations de grain.
  Le générateur **ignore en grande partie le bruit** : c'est l'effondrement de modes annoncé plus haut.

### Interprétation
- **Le GAN fait mieux que Telea, mais de peu.** Il a appris « de quoi est fait » un zellige (couleurs, grain), mais pas
  **comment il est construit** (étoiles, symétries, lignes continues). Les motifs du zellige sont grands par rapport
  au trou et obéissent à une géométrie globale, qu'un petit U-Net entraîné 18 min sur 697 images ne capte pas.
- **Pourquoi le PSNR reste bas (≈ 12 dB), même pour une complétion plausible.** Le PSNR compare chaque pixel à celui
  de l'original, au même endroit. Imaginons un modèle qui dessinerait une étoile parfaite, mais **décalée** de
  quelques pixels : là où l'original a une ligne blanche, il a du brun, et inversement. Sur un zellige très contrasté
  (lignes beige clair sur fond brun foncé ou noir), chaque pixel décalé coûte une erreur énorme, de l'ordre de 150 à 200
  sur 255. Cette étoile serait donc **plus pénalisée** qu'une bouillie grise moyenne, qui reste « à mi-chemin » de tout.
  Le PSNR récompense la **prudence** (la moyenne), pas la **plausibilité**. C'est aussi pour cela que la L1 pousse
  le GAN vers un grain moyen sans forme franche.
- **Ce que cela implique :** les écarts de PSNR entre méthodes resteront faibles. Il faut ajouter **LPIPS** (perception)
  et s'appuyer sur les **exemples visuels** pour juger la plausibilité.
- **Diversité faible :** c'est la limite principale vis-à-vis de l'objectif (proposer 4 complétions différentes).
  Elle motive le **VQ-VAE + PixelCNN**, qui **tire au hasard** chaque morceau de l'image dans une distribution apprise :
  la diversité fait alors partie de la méthode, au lieu de dépendre d'un bruit que le réseau peut ignorer.

### Pistes pour une version v2 (si le temps le permet)
- Entraîner plus longtemps : la L1 n'a pas atteint de plateau.
- Choisir le meilleur modèle sur le SSIM ou LPIPS plutôt que sur le PSNR, qui oscille trop.
- Encourager la diversité : perte de « mode seeking » (pénaliser deux bruits différents qui donnent la même image)
  ou bruit injecté à plusieurs niveaux du U-Net.
- Améliorer le raccord : discriminateur qui juge aussi l'image entière, ou couches d'attention pour aller chercher
  des motifs ailleurs dans l'image.

### Questions probables du jury
1. **« Pourquoi deux pertes (L1 et adverse) et pas une seule ? »**
   Avec seulement la L1, le réseau apprend à produire la **moyenne** de toutes les possibilités, ce qui donne une zone floue
   (comme Telea, en mieux). Avec seulement la perte adverse, il produit des motifs nets mais sans lien avec l'image.
   Ensemble : fidèle **et** net. Le poids 100 pour la L1 vient du modèle de référence pix2pix.
2. **« Vos 4 propositions se ressemblent beaucoup. Pourquoi ? »**
   C'est exact, et je l'ai mesuré : elles ne diffèrent que de 7 à 10 sur 255, soit environ 5 fois moins qu'elles ne diffèrent
   de l'original. Le générateur a appris à **ignorer le bruit**, car la perte L1 le pousse vers une seule réponse « moyenne »,
   la plus sûre. C'est l'effondrement de modes, un défaut bien connu des GAN conditionnels. C'est la raison principale
   du passage au VQ-VAE + PixelCNN, où la diversité vient d'un vrai tirage au hasard.
3. **« Le GAN ne gagne que 0,5 dB. Est-ce que ça valait la peine ? »**
   Le gain est faible mais **systématique** (9 configurations sur 9), et le GAN est plus rapide (1,9 ms contre 11 ms par image),
   pour seulement 2 Go de GPU. Surtout, le PSNR sous-estime l'écart réel : il pénalise fortement un motif net mais décalé,
   et favorise le flou. Visuellement, le GAN reprend la bonne texture et les bonnes couleurs, ce que Telea ne fait pas.
   Mais il ne reconstruit pas encore la géométrie (pas d'étoiles, raccord visible). Je présente donc ce résultat honnêtement,
   comme une amélioration partielle. La L1 encore en baisse à l'epoch 150 montre qu'un entraînement plus long aiderait.

### GAN v2 : ce qu'on change et pourquoi

> Statut : code prêt et testé sur CPU (une itération). **Entraînement sur Kaggle à lancer**
> (`notebooks/03b_gan_v2.ipynb`, config `configs/gan_v2.yaml`). Pas encore de résultats.

La v2 s'attaque aux défauts observés sur la v1. **Une seule architecture change** ; tout le reste (données, masques,
pertes, poids L1 = 100, taux d'apprentissage, graine 42, protocole de test) est identique, pour que la comparaison soit juste.
La v1 n'est pas modifiée : son code, sa config et ses résultats restent valables.

| Changement | Défaut v1 visé | Pourquoi ça devrait aider |
|---|---|---|
| **Blocs résiduels à convolutions dilatées** (dilatations 2, 4, 8, 16) au niveau 16×16 du U-Net | Pas d'étoiles, pas de lignes qui se prolongent, raccord visible | Pour dessiner une étoile au centre d'un grand trou, le réseau doit « voir » les morceaux d'étoiles **de l'autre côté du trou**. Les blocs dilatés agrandissent ce qu'il voit (son champ réceptif) sans réduire encore l'image. |
| **300 epochs** au lieu de 150 | La L1 baissait encore à l'epoch 150 | Le modèle v1 n'avait pas fini d'apprendre. |
| **Sélection du meilleur modèle sur le SSIM** au lieu du PSNR | Le PSNR de validation oscillait de ±0,2 dB sans tendance ; l'epoch 110 était en partie due au hasard | Le SSIM de validation montait régulièrement : c'est un critère plus stable, et plus proche de la perception de la structure. |

#### Notions clés
- **Champ réceptif :** la zone de l'image d'entrée qu'un « neurone » du réseau peut voir. Si elle est plus petite que le trou,
  le centre du trou est rempli « à l'aveugle », sans information venant des bords : c'est ce qui donne le grain uniforme de la v1.
- **Convolution dilatée :** un filtre 3×3 normal regarde 9 pixels voisins collés. Un filtre dilaté de 4 regarde aussi 9 points,
  mais **espacés de 4 pixels**. Il voit donc une zone de 9×9 avec le même nombre de paramètres. En enchaînant des dilatations
  2, 4 et 8, la vue s'élargit très vite.
- **Bloc résiduel :** le bloc calcule une **correction** qu'on ajoute à son entrée (sortie = entrée + correction), au lieu
  de tout recalculer. Au début de l'entraînement, il laisse passer l'information sans l'abîmer, ce qui rend stable l'empilement de plusieurs blocs.
- **Pourquoi au niveau 16×16 :** à ce niveau, chaque case résume un carré de 8×8 pixels de l'image. Une dilatation de 8
  y couvre donc une grande partie de l'image. Plus bas (8×8, 4×4), l'image est trop réduite pour garder la position des motifs.

#### Ce que ça coûte (mesuré sur CPU, batch 32)
- Paramètres du générateur : **10,07 M** au lieu de 5,34 M (+4,72 M, soit ×1,9). Discriminateur inchangé (2,77 M).
- Mémoire des activations du générateur : **1,33 Go** au lieu de 1,12 Go (+210 Mo). Poids + optimiseur Adam : 0,20 Go au lieu de 0,13 Go.
- On peut donc s'attendre à un pic d'environ **2,3 Go** sur GPU (v1 mesurée : 2,03 Go), bien sous la limite de 12 Go.
  La valeur réelle sera relevée dans `memoire_gpu_gan_v2.json`.
- Temps : environ 2 fois plus d'epochs et un réseau un peu plus lourd, donc **environ 40 min** à prévoir (v1 : 18 min).

#### Point d'attention : la dilatation 16
Sur une carte de 16×16 cases, un filtre dilaté de 16 place ses 8 points extérieurs **à 16 cases du centre, donc toujours
en dehors de la carte**, dans le remplissage à zéro. Il ne voit donc que la case centrale et se comporte comme un simple
mélange de canaux (convolution 1×1). Je l'ai vérifié numériquement. Ce bloc n'est pas nuisible, mais il n'élargit pas la vue.
Les dilatations 2, 4 et 8 suffisent déjà : à elles trois, elles couvrent toute la carte 16×16, donc toute l'image.
J'ai gardé 2, 4, 8, 16 comme prévu. La liste est réglable dans `configs/gan_v2.yaml` (`dilatations`) si l'on veut tester [1, 2, 4, 8].

#### Ce que la v2 ne corrige pas
- **La diversité.** Rien dans la v2 n'oblige le générateur à utiliser le bruit, donc les 4 propositions risquent de rester
  presque identiques. C'est un choix délibéré : on change **une chose à la fois** pour savoir ce qui a un effet.
  La diversité est le rôle du VQ-VAE + PixelCNN.

#### Questions probables du jury (v2)
1. **« Pourquoi des convolutions dilatées plutôt qu'un réseau plus profond ? »**
   Réduire encore l'image (8×8, 4×4) fait perdre la position précise des motifs. La dilatation élargit la vue **sans réduire l'image**
   et avec peu de paramètres en plus. C'est une technique standard en inpainting (par exemple dans le modèle de référence d'Iizuka et al., 2017).
2. **« Pourquoi avoir changé trois choses en même temps ? Comment savoir laquelle a aidé ? »**
   Une seule touche le modèle (les blocs dilatés). Les deux autres corrigent la façon d'entraîner : la durée, parce que la v1
   n'avait pas fini d'apprendre, et le critère de choix, parce que le PSNR était trop bruité. Pour isoler l'effet des blocs dilatés,
   il faudrait aussi entraîner une v1 sur 300 epochs. C'est une limite que j'assume, faute de temps de calcul.
3. **« Pourquoi choisir le meilleur modèle sur le SSIM et pas sur le PSNR ? »**
   Sur la v1, le PSNR de validation sautait de ±0,2 dB d'une évaluation à l'autre sans tendance, alors que le SSIM montait
   régulièrement. Choisir sur une mesure aussi bruitée revient en partie à tirer au sort. Le SSIM est plus stable et mesure la structure.

## 5) VQ-VAE + PixelCNN — *à compléter*

## 6) Comparaison finale (PSNR, SSIM, LPIPS, diversité, temps, mémoire GPU) — *à compléter*

## 7) Démo Gradio — *à compléter*

## 8) Éthique et limites — *à compléter*
