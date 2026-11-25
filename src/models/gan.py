"""GAN d'inpainting : générateur U-Net compact + discriminateur PatchGAN.

Générateur
  Entrée : 5 canaux = image trouée (3) + masque (1) + bruit (1).
  Sortie : RGB en tanh, puis composition :
           pixels connus de l'entrée + pixels générés uniquement dans le trou.
  Le bruit permet d'obtenir des propositions différentes pour une même image.

Discriminateur
  PatchGAN : au lieu d'un seul « vrai / faux » pour toute l'image, il donne une note
  par zone (patch) de l'image. Normalisation spectrale pour stabiliser l'entraînement.
  Il reçoit l'image + le masque (4 canaux) pour savoir où se trouve le trou.
"""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


# ----------------------------------------------------------------------------- Générateur

class BlocDescente(nn.Module):
    """Divise la taille par 2 (convolution de pas 2)."""

    def __init__(self, c_in, c_out, norm=True):
        super().__init__()
        couches = [nn.Conv2d(c_in, c_out, 4, 2, 1, bias=not norm)]
        if norm:
            couches.append(nn.BatchNorm2d(c_out))
        couches.append(nn.LeakyReLU(0.2, inplace=True))
        self.bloc = nn.Sequential(*couches)

    def forward(self, x):
        return self.bloc(x)


class BlocMontee(nn.Module):
    """Multiplie la taille par 2 (agrandissement + convolution, évite les artefacts en damier)."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.bloc = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.bloc(x)


class BlocResiduelDilate(nn.Module):
    """Bloc résiduel à convolution dilatée (v2).

    La convolution dilatée « écarte » les points regardés par le filtre (de `dilatation` pixels) :
    même nombre de paramètres, mais elle voit beaucoup plus loin. Le raccourci résiduel (x + ...)
    permet d'empiler ces blocs sans gêner l'apprentissage.
    """

    def __init__(self, c, dilatation):
        super().__init__()
        self.bloc = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, padding=dilatation, dilation=dilatation, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.bloc(x))


class Generateur(nn.Module):
    """U-Net compact : 128 -> 64 -> 32 -> 16 -> 8 -> 4 puis remontée avec connexions de saut.

    v1 : Generateur(base=64)                         (comportement d'origine, inchangé)
    v2 : Generateur(base=64, dilatation=True)        (blocs dilatés au niveau 16x16)
    """

    def __init__(self, base=64, dilatation=False, dilatations=(1, 2, 4, 8)):
        super().__init__()
        b = base
        self.d1 = BlocDescente(5, b, norm=False)     # 64x64
        self.d2 = BlocDescente(b, 2 * b)             # 32x32
        self.d3 = BlocDescente(2 * b, 4 * b)         # 16x16
        self.d4 = BlocDescente(4 * b, 4 * b)         # 8x8
        self.d5 = BlocDescente(4 * b, 4 * b)         # 4x4 (goulot)

        self.u1 = BlocMontee(4 * b, 4 * b)           # 8x8
        self.u2 = BlocMontee(8 * b, 4 * b)           # 16x16 (entrée = u1 + d4)
        self.u3 = BlocMontee(8 * b, 2 * b)           # 32x32 (u2 + d3)
        self.u4 = BlocMontee(4 * b, b)               # 64x64 (u3 + d2)
        self.u5 = BlocMontee(2 * b, b)               # 128x128 (u4 + d1)
        self.sortie = nn.Sequential(nn.Conv2d(b, 3, 3, 1, 1), nn.Tanh())

        # v2 : créés en dernier et seulement si demandé -> la v1 garde exactement
        # les mêmes couches, la même initialisation et les mêmes clés de sauvegarde.
        self.dilate = None
        if dilatation:
            self.dilate = nn.Sequential(*[BlocResiduelDilate(4 * b, d) for d in dilatations])

    def forward(self, image_trouee, masque, bruit=None):
        """Renvoie (image_finale, sortie_brute).

        image_finale = pixels connus + pixels générés dans le trou.
        """
        if bruit is None:
            bruit = torch.randn_like(masque)
        x = torch.cat([image_trouee, masque, bruit], dim=1)

        e1 = self.d1(x)
        e2 = self.d2(e1)
        e3 = self.d3(e2)
        if self.dilate is not None:
            e3 = self.dilate(e3)                     # v2 : contexte élargi, utilisé ensuite partout
        e4 = self.d4(e3)
        e5 = self.d5(e4)

        y = self.u1(e5)
        y = self.u2(torch.cat([y, e4], 1))
        y = self.u3(torch.cat([y, e3], 1))
        y = self.u4(torch.cat([y, e2], 1))
        y = self.u5(torch.cat([y, e1], 1))
        brute = self.sortie(y)

        finale = image_trouee * (1 - masque) + brute * masque
        return finale, brute


# ----------------------------------------------------------------------------- Discriminateur

class Discriminateur(nn.Module):
    """PatchGAN avec normalisation spectrale. Entrée 128x128 -> carte de notes 14x14."""

    def __init__(self, base=64):
        super().__init__()
        b = base

        def conv(c_in, c_out, pas):
            return spectral_norm(nn.Conv2d(c_in, c_out, 4, pas, 1))

        self.reseau = nn.Sequential(
            conv(4, b, 2), nn.LeakyReLU(0.2, inplace=True),            # 64x64
            conv(b, 2 * b, 2), nn.LeakyReLU(0.2, inplace=True),        # 32x32
            conv(2 * b, 4 * b, 2), nn.LeakyReLU(0.2, inplace=True),    # 16x16
            conv(4 * b, 8 * b, 1), nn.LeakyReLU(0.2, inplace=True),    # 15x15
            conv(8 * b, 1, 1),                                         # 14x14
        )

    def forward(self, image, masque):
        return self.reseau(torch.cat([image, masque], dim=1))


def creer_generateur(cfg_modele):
    """Construit le générateur à partir de la section `modele` d'une config (v1 ou v2)."""
    return Generateur(cfg_modele["base_generateur"],
                      dilatation=cfg_modele.get("dilatation", False),
                      dilatations=tuple(cfg_modele.get("dilatations", (1, 2, 4, 8))))


def compter_parametres(modele):
    return sum(p.numel() for p in modele.parameters())
