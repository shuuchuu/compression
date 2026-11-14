from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

TAILLE_IMAGE = (88, 88)


def cree_transformation(taille_image: tuple[int, int]) -> v2.Compose:
    return v2.Compose(
        [v2.ToImage(), v2.Resize(taille_image), v2.ToDtype(torch.float32, scale=True)]
    )


def cree_chargeur(
    dossier: str, taille_image: tuple[int, int], melange: bool = False
) -> DataLoader:
    jeu = datasets.ImageFolder(dossier, transform=cree_transformation(taille_image))
    return DataLoader(jeu, batch_size=64, shuffle=melange)


def prepare(
    dossier_ent: str, dossier_test: str, images_par_classe: int, dossier_prepare: str
) -> None:
    for partie, dossier in (("entrainement", dossier_ent), ("test", dossier_test)):
        for dossier_classe in sorted(Path(dossier).iterdir()):
            dossier_sortie = Path(dossier_prepare) / partie / dossier_classe.name
            dossier_sortie.mkdir(parents=True, exist_ok=True)
            for chemin in sorted(dossier_classe.iterdir())[:images_par_classe]:
                image = Image.open(chemin).convert("RGB").resize(TAILLE_IMAGE)
                image.save(dossier_sortie / chemin.name, quality=95)
