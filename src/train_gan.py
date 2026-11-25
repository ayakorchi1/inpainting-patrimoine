"""Entraînement du GAN d'inpainting.

Utilisation (Kaggle) :
    python -m src.train_gan --config configs/gan.yaml --npz /kaggle/input/.../zellige_128.npz
ou depuis un notebook :
    from src.train_gan import entrainer
    entrainer("configs/gan.yaml", chemin_npz)

Pertes
  Générateur     : poids_l1 * L1 dans le trou  +  poids_adverse * hinge (-D(faux))
  Discriminateur : hinge = relu(1 - D(vrai)) + relu(1 + D(faux))

Fichiers produits dans `sorties.dossier` (par défaut /kaggle/working, jamais dans Git) :
  v1 (configs/gan.yaml, sans `experience`)   v2 (configs/gan_v2.yaml, experience: gan_v2)
  gan_meilleur.pt                             gan_v2_meilleur.pt       meilleur générateur (critère PSNR ou SSIM)
  gan_dernier.pt                              gan_v2_dernier.pt        générateur en fin d'entraînement
  historique_gan.csv                          historique_gan_v2.csv    pertes par epoch + PSNR/SSIM de validation
  memoire_gpu.json                            memoire_gpu_gan_v2.json  pic de mémoire GPU (limite 12 Go)
"""
import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.data import ZelligeDataset, vers_tenseur, vers_uint8
from src.masks import creer_masque
from src.metrics import psnr_trou, ssim_img
from src.models.gan import Discriminateur, compter_parametres, creer_generateur


def fixer_graine(graine):
    random.seed(graine)
    np.random.seed(graine)
    torch.manual_seed(graine)
    torch.cuda.manual_seed_all(graine)


def charger_config(chemin):
    with open(chemin, encoding="utf-8") as f:
        return yaml.safe_load(f)


def l1_trou(sortie, cible, masque):
    """Erreur absolue moyenne calculée uniquement sur les pixels du trou."""
    return (torch.abs(sortie - cible) * masque).sum() / (masque.sum() * 3 + 1e-8)


@torch.no_grad()
def generer(G, image_trouee, masque, graine=None):
    """Complétion avec un bruit reproductible si `graine` est donnée."""
    if graine is None:
        bruit = torch.randn_like(masque)
    else:
        gen = torch.Generator(device=masque.device).manual_seed(graine)
        bruit = torch.randn(masque.shape, generator=gen, device=masque.device)
    finale, _ = G(image_trouee, masque, bruit)
    return finale


@torch.no_grad()
def completer(G, X, masques, graine=0, batch=32, device="cpu"):
    """Complète des images uint8 (N, H, W, 3) avec des masques (N, H, W). Renvoie uint8 (N, H, W, 3).

    Même `graine` -> mêmes propositions ; graine différente -> autre proposition.
    """
    G.eval()
    sorties = []
    for i in range(0, len(X), batch):
        img = torch.stack([vers_tenseur(x) for x in X[i:i + batch]]).to(device)
        m = torch.from_numpy(np.asarray(masques[i:i + batch])).float().unsqueeze(1).to(device)
        sorties.append(vers_uint8(generer(G, img * (1 - m), m, graine + i)))
    return np.concatenate(sorties)


@torch.no_grad()
def evaluer(G, loader, device, graine_bruit=0):
    """PSNR (dans le trou) et SSIM moyens, calculés avec src/metrics.py comme la baseline."""
    G.eval()
    psnrs, ssims = [], []
    for img, trouee, m in loader:
        img, trouee, m = img.to(device), trouee.to(device), m.to(device)
        rest = vers_uint8(generer(G, trouee, m, graine_bruit))
        orig = vers_uint8(img)
        masques = m[:, 0].cpu().numpy().astype(np.uint8)
        for o, r, mm in zip(orig, rest, masques):
            psnrs.append(psnr_trou(o, r, mm))
            ssims.append(ssim_img(o, r))
    G.train()
    return float(np.mean(psnrs)), float(np.mean(ssims))


def noms_fichiers(experience=None):
    """Noms des fichiers produits.

    Sans nom d'expérience (config v1) : noms d'origine, pour que le notebook 03 reste valable.
    Avec un nom (ex. "gan_v2")       : gan_v2_meilleur.pt, historique_gan_v2.csv, memoire_gpu_gan_v2.json...
    """
    if experience is None:
        return {"meilleur": "gan_meilleur.pt", "dernier": "gan_dernier.pt",
                "historique": "historique_gan.csv", "memoire": "memoire_gpu.json"}
    return {"meilleur": f"{experience}_meilleur.pt", "dernier": f"{experience}_dernier.pt",
            "historique": f"historique_{experience}.csv", "memoire": f"memoire_gpu_{experience}.json"}


def charger_generateur(chemin_pt, device="cpu"):
    """Recharge un générateur sauvegardé (v1 ou v2) en mode évaluation. Renvoie (G, checkpoint)."""
    ckpt = torch.load(chemin_pt, map_location=device, weights_only=False)
    G = creer_generateur(ckpt["config"]["modele"]).to(device)
    G.load_state_dict(ckpt["generateur"])
    G.eval()
    return G, ckpt


def entrainer(chemin_config, chemin_npz, dossier=None, epochs=None, max_iters=None, device=None):
    """Lance l'entraînement. `epochs`, `max_iters` et `dossier` remplacent la config (utile pour tester).

    Options de config (facultatives, valeurs par défaut = v1) :
      experience                    nom utilisé dans les fichiers produits (défaut : noms d'origine)
      evaluation.critere_selection  "psnr" (défaut) ou "ssim" pour choisir le meilleur modèle
    """
    cfg = charger_config(chemin_config)
    ent, ev = cfg["entrainement"], cfg["evaluation"]
    noms = noms_fichiers(cfg.get("experience"))
    critere = ev.get("critere_selection", "psnr")
    if critere not in ("psnr", "ssim"):
        raise ValueError(f"critere_selection doit être 'psnr' ou 'ssim', pas {critere!r}")
    epochs = epochs or ent["epochs"]
    dossier = dossier or cfg["sorties"]["dossier"]
    os.makedirs(dossier, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    fixer_graine(cfg["graine"])

    # --- Données
    train = ZelligeDataset(chemin_npz, "train")
    val_imgs = np.load(chemin_npz)["val"]
    rng = np.random.default_rng(cfg["graine"])
    H = cfg["donnees"]["taille"]
    masques_val = np.stack([creer_masque(ev["masque_type"], ev["masque_ratio"], rng, H)
                            for _ in range(len(val_imgs))])
    val = ZelligeDataset(chemin_npz, "val", masques=masques_val)
    nw = cfg["donnees"]["num_workers"]
    loader = DataLoader(train, batch_size=ent["batch"], shuffle=True, drop_last=True,
                        num_workers=nw, pin_memory=(device == "cuda"))
    loader_val = DataLoader(val, batch_size=ent["batch"], shuffle=False, num_workers=nw)

    # --- Modèles et optimiseurs
    G = creer_generateur(cfg["modele"]).to(device)
    D = Discriminateur(cfg["modele"]["base_discriminateur"]).to(device)
    print(f"Paramètres : G = {compter_parametres(G)/1e6:.2f} M | D = {compter_parametres(D)/1e6:.2f} M")
    betas = (ent["beta1"], ent["beta2"])
    opt_G = torch.optim.Adam(G.parameters(), lr=ent["lr"], betas=betas)
    opt_D = torch.optim.Adam(D.parameters(), lr=ent["lr"], betas=betas)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    historique, iters = [], 0
    meilleur = {"score": -np.inf, "epoch": None, "psnr_val": None, "ssim_val": None}
    t_debut = time.time()
    for epoch in range(1, epochs + 1):
        sommes = dict(perte_D=0.0, perte_G=0.0, l1=0.0, adverse=0.0)
        n = 0
        for img, trouee, m in loader:
            img, trouee, m = img.to(device), trouee.to(device), m.to(device)
            finale, brute = G(trouee, m)

            # 1) Discriminateur : vraies images -> note haute, fausses -> note basse (hinge)
            perte_D = F.relu(1 - D(img, m)).mean() + F.relu(1 + D(finale.detach(), m)).mean()
            opt_D.zero_grad(set_to_none=True)
            perte_D.backward()
            opt_D.step()

            # 2) Générateur : être fidèle dans le trou (L1) + tromper le discriminateur
            l1 = l1_trou(brute, img, m)
            adverse = -D(finale, m).mean()
            perte_G = ent["poids_l1"] * l1 + ent["poids_adverse"] * adverse
            opt_G.zero_grad(set_to_none=True)
            perte_G.backward()
            opt_G.step()

            for k, v in zip(sommes, (perte_D, perte_G, l1, adverse)):
                sommes[k] += v.item()
            n += 1
            iters += 1
            if max_iters and iters >= max_iters:
                break

        ligne = {"epoch": epoch, **{k: v / max(n, 1) for k, v in sommes.items()},
                 "psnr_val": np.nan, "ssim_val": np.nan, "minutes": (time.time() - t_debut) / 60}

        fin = epoch == epochs or (max_iters and iters >= max_iters)
        if epoch % ev["tous_les"] == 0 or fin:
            psnr_v, ssim_v = evaluer(G, loader_val, device, ev["graine_bruit"])
            ligne.update(psnr_val=psnr_v, ssim_val=ssim_v)
            score = psnr_v if critere == "psnr" else ssim_v
            if score > meilleur["score"]:
                meilleur = {"score": score, "epoch": epoch, "psnr_val": psnr_v, "ssim_val": ssim_v}
                torch.save({"generateur": G.state_dict(), "config": cfg, "epoch": epoch,
                            "psnr_val": psnr_v, "ssim_val": ssim_v, "critere_selection": critere},
                           os.path.join(dossier, noms["meilleur"]))
            print(f"[epoch {epoch:3d}] D {ligne['perte_D']:.3f} | G {ligne['perte_G']:.3f} "
                  f"| L1 {ligne['l1']:.4f} | val PSNR {psnr_v:.2f} dB SSIM {ssim_v:.3f}")

        historique.append(ligne)
        pd.DataFrame(historique).to_csv(os.path.join(dossier, noms["historique"]), index=False)
        if fin:
            break

    torch.save({"generateur": G.state_dict(), "config": cfg, "epoch": epoch},
               os.path.join(dossier, noms["dernier"]))

    # --- Mémoire GPU
    pic_go = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0
    infos = {"pic_memoire_gpu_Go": round(pic_go, 3), "limite_Go": 12, "respecte_limite": pic_go <= 12,
             "device": device, "minutes": round((time.time() - t_debut) / 60, 2),
             "experience": cfg.get("experience", "gan"),
             "parametres_G_M": round(compter_parametres(G) / 1e6, 2),
             "critere_selection": critere, "meilleur_epoch": meilleur["epoch"],
             "meilleur_psnr_val": meilleur["psnr_val"], "meilleur_ssim_val": meilleur["ssim_val"]}
    with open(os.path.join(dossier, noms["memoire"]), "w") as f:
        json.dump(infos, f, indent=2)
    print(f"Pic mémoire GPU : {pic_go:.2f} Go (limite 12 Go) | durée {infos['minutes']} min")
    return pd.DataFrame(historique), infos


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/gan.yaml")
    p.add_argument("--npz", required=True)
    p.add_argument("--dossier", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max_iters", type=int, default=None)
    a = p.parse_args()
    entrainer(a.config, a.npz, a.dossier, a.epochs, a.max_iters)
