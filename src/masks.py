"""Génération des masques d'inpainting.

Convention : masque uint8 de taille (H, H), 1 = pixel effacé, 0 = pixel connu.
Logique identique à notebooks/01_exploration_preparation.ipynb.
"""
import cv2
import numpy as np

TAILLE = 128
TYPES = ["central", "carre", "irregulier"]


def masque_central(ratio, H=TAILLE):
    """Carré centré couvrant `ratio` de l'image."""
    c = int(round(H * np.sqrt(ratio)))          # côté du carré pour couvrir `ratio` de l'image
    m = np.zeros((H, H), np.uint8)
    d = (H - c) // 2
    m[d:d+c, d:d+c] = 1
    return m


def masque_carre(ratio, rng, H=TAILLE):
    """Carré placé au hasard couvrant `ratio` de l'image."""
    c = int(round(H * np.sqrt(ratio)))
    x, y = rng.integers(0, H - c + 1, size=2)    # position aléatoire
    m = np.zeros((H, H), np.uint8)
    m[y:y+c, x:x+c] = 1
    return m


def masque_irregulier(ratio, rng, H=TAILLE):
    """Traits épais aléatoires (type « fissures ») jusqu'à couvrir au moins `ratio`."""
    m = np.zeros((H, H), np.uint8)
    while m.mean() < ratio:                      # on ajoute des traits jusqu'à atteindre la surface voulue
        x, y = rng.integers(0, H, size=2)
        for _ in range(rng.integers(4, 10)):
            angle = rng.uniform(0, 2 * np.pi)
            L = rng.integers(10, 30)
            x2 = int(np.clip(x + L * np.cos(angle), 0, H - 1))
            y2 = int(np.clip(y + L * np.sin(angle), 0, H - 1))
            cv2.line(m, (int(x), int(y)), (x2, y2), 1, int(rng.integers(8, 16)))
            x, y = x2, y2
            if m.mean() >= ratio:
                break
    return m


def creer_masque(type_, ratio, rng, H=TAILLE):
    """Crée un masque du type demandé ("central", "carre" ou "irregulier")."""
    if type_ == "central":
        return masque_central(ratio, H)
    if type_ == "carre":
        return masque_carre(ratio, rng, H)
    return masque_irregulier(ratio, rng, H)


def masque_aleatoire(rng, H=TAILLE, ratio_min=0.05, ratio_max=0.45):
    """Masque pour l'entraînement : type tiré au hasard, taille entre 5 et 45 %."""
    type_ = TYPES[rng.integers(len(TYPES))]
    ratio = rng.uniform(ratio_min, ratio_max)
    return creer_masque(type_, ratio, rng, H)
