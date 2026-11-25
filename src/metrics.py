"""Métriques d'évaluation.

Identiques à notebooks/02-baseline-opencv (1).ipynb.
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
