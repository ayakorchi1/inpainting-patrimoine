"""Métriques d'évaluation.

Identiques à notebooks/02_baseline_opencv.ipynb.
Images attendues : uint8 (H, W, 3), valeurs 0-255.
"""
import numpy as np
from skimage.metrics import structural_similarity as ssim


def psnr_trou(orig, rest, m):
    """PSNR (dB) calculé uniquement sur les pixels du trou (m == 1)."""
    diff = (orig.astype(np.float32) - rest.astype(np.float32))[m == 1]
    mse = np.mean(diff ** 2)
    return 10 * np.log10(255 ** 2 / mse) if mse > 0 else 100


def ssim_img(orig, rest):
    """SSIM sur l'image entière (couleur)."""
    return ssim(orig, rest, channel_axis=2, data_range=255)


def diversite_trou(propositions, m):
    """Écart moyen entre plusieurs propositions pour une même image, dans le trou.

    propositions : liste de K images uint8 (H, W, 3) ; m : masque (H, W), 1 = trou.
    Renvoie la différence absolue moyenne (échelle 0-255) sur toutes les paires.
    0 = propositions identiques (aucune diversité).
    """
    ecarts = []
    for a in range(len(propositions)):
        for b in range(a + 1, len(propositions)):
            diff = np.abs(propositions[a].astype(np.float32) - propositions[b].astype(np.float32))
            ecarts.append(diff[m == 1].mean())
    return float(np.mean(ecarts))
