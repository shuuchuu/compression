import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .donnees import TAILLE_IMAGE, cree_chargeur
from .modele import LeNet

CHEMIN_METRIQUES = Path("metriques/entrainement.json")


def entraine_epoque(
    modele: nn.Module, chargeur: DataLoader, optimiseur: torch.optim.Optimizer
) -> float:
    critere = nn.CrossEntropyLoss()
    modele.train()
    total, nombre = 0.0, 0
    for images, etiquettes in chargeur:
        optimiseur.zero_grad()
        perte = critere(modele(images), etiquettes)
        perte.backward()
        optimiseur.step()
        total += perte.item() * len(etiquettes)
        nombre += len(etiquettes)
    return total / nombre


def evalue(modele: nn.Module, chargeur: DataLoader) -> float:
    modele.eval()
    corrects, nombre = 0, 0
    with torch.inference_mode():
        for images, etiquettes in chargeur:
            corrects += int((modele(images).argmax(dim=1) == etiquettes).sum())
            nombre += len(etiquettes)
    return corrects / nombre


def entraine(
    dossier_prepare: str, taux_apprentissage: float, nb_epoques: int, chemin_modele: str
) -> None:
    torch.manual_seed(0)
    chargeur_ent = cree_chargeur(
        f"{dossier_prepare}/entrainement", TAILLE_IMAGE, melange=True
    )
    chargeur_test = cree_chargeur(f"{dossier_prepare}/test", TAILLE_IMAGE)
    modele = LeNet()
    optimiseur = torch.optim.AdamW(modele.parameters(), lr=taux_apprentissage)
    for epoque in range(nb_epoques):
        perte = entraine_epoque(modele, chargeur_ent, optimiseur)
        print(f"époque {epoque + 1}/{nb_epoques} : perte={perte:.3f}")
    exactitude = evalue(modele, chargeur_test)
    print(f"exactitude={exactitude:.3f}")
    torch.save(modele.state_dict(), chemin_modele)
    CHEMIN_METRIQUES.parent.mkdir(exist_ok=True)
    CHEMIN_METRIQUES.write_text(json.dumps({"exactitude": exactitude, "perte": perte}))
