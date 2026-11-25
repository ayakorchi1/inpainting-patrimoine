"""Chargement des données pour PyTorch.

- Lit zellige_128.npz (clés "train", "val", "test", uint8 (N, 128, 128, 3)).
- Normalise les images en [-1, 1] (adapté à la sortie tanh du générateur).
- Augmentation (retournements + rotations de 90°) seulement sur "train".
- Associe à chaque image un masque (1 = pixel effacé) :
    * par défaut, un masque aléatoire NOUVEAU à chaque appel (src.masks.masque_aleatoire) ;
    * ou des masques fixes passés en paramètre (validation / test).
"""
import glob

import numpy as np
import torch
from torch.utils.data import Dataset

from src.masks import masque_aleatoire


def trouver_npz(nom, racine="/kaggle/input"):
    """Cherche un fichier .npz (ex. "zellige_128.npz") n'importe où sous `racine`."""
    trouves = glob.glob(f"{racine}/**/{nom}", recursive=True)
    if not trouves:
        raise FileNotFoundError(f"{nom} introuvable sous {racine}")
    return trouves[0]


def vers_tenseur(img_uint8):
    """uint8 (H, W, 3) -> float (3, H, W) dans [-1, 1]."""
    return torch.from_numpy(img_uint8).permute(2, 0, 1).float() / 127.5 - 1.0


def vers_uint8(t):
    """float (B, 3, H, W) ou (3, H, W) dans [-1, 1] -> uint8 (B, H, W, 3) ou (H, W, 3)."""
    t = ((t.detach().cpu().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return t.permute(0, 2, 3, 1).numpy() if t.dim() == 4 else t.permute(1, 2, 0).numpy()


class ZelligeDataset(Dataset):
    """Renvoie (image, image_trouee, masque).

    image        : (3, H, W) dans [-1, 1]
    image_trouee : (3, H, W), les pixels du trou sont mis à 0 (gris moyen)
    masque       : (1, H, W), 1 = pixel effacé
    """

    def __init__(self, chemin_npz, split="train", masques=None, augmenter=None):
        self.images = np.load(chemin_npz)[split]
        self.masques = masques                           # None = masques aléatoires
        self.augmenter = (split == "train") if augmenter is None else augmenter

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = vers_tenseur(self.images[i])

        if self.augmenter:
            if torch.rand(1) < 0.5:
                img = img.flip(2)                        # retournement horizontal
            if torch.rand(1) < 0.5:
                img = img.flip(1)                        # retournement vertical
            k = int(torch.randint(0, 4, (1,)))
            img = torch.rot90(img, k, dims=(1, 2))       # rotation de 0, 90, 180 ou 270°

        if self.masques is not None:
            m = self.masques[i]
        else:
            # Graine tirée par torch : différente à chaque appel ET dans chaque worker du DataLoader.
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))
            m = masque_aleatoire(rng, H=img.shape[1])
        m = torch.from_numpy(np.ascontiguousarray(m)).float().unsqueeze(0)

        image_trouee = img * (1 - m)
        return img, image_trouee, m
