import gzip
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader

from .donnees import TAILLE_IMAGE, cree_chargeur
from .entrainement import entraine_epoque
from .modele import LeNet

CHEMIN_METRIQUES = Path("metriques/compression.json")


def exporte_onnx(modele: nn.Module, chemin: str) -> None:
    exemple = (torch.rand(1, 3, *TAILLE_IMAGE),)
    torch.onnx.export(
        modele.eval(), exemple, chemin, dynamo=True, external_data=False, verbose=False
    )


class Calibration(CalibrationDataReader):
    def __init__(self, images: list[torch.Tensor]) -> None:
        self.images = iter(images)

    def get_next(self) -> dict | None:  # pyrefly: ignore[bad-override]
        image = next(self.images, None)
        return None if image is None else {"x": image.numpy()}


def mesure(chemin: str, images: np.ndarray, etiquettes: np.ndarray) -> dict:
    session = ort.InferenceSession(chemin, providers=["CPUExecutionProvider"])
    predictions, durees = [], []
    for image in images:
        debut = time.perf_counter()
        (logits,) = session.run(None, {"x": image[None]})
        durees.append(time.perf_counter() - debut)
        predictions.append(logits.argmax())  # pyrefly: ignore[missing-attribute]
    return {
        "exactitude": float(np.mean(np.array(predictions) == etiquettes)),
        "taille_mo": Path(chemin).stat().st_size / 1e6,
        "gzip_mo": len(gzip.compress(Path(chemin).read_bytes())) / 1e6,
        "latence_ms": float(1000 * np.median(durees[10:])),
    }


def rassemble_images(chargeur: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    images, etiquettes = zip(*chargeur, strict=True)
    return torch.cat(images).numpy(), torch.cat(etiquettes).numpy()


def compresse(
    dossier_prepare: str,
    chemin_modele: str,
    taux_apprentissage: float,
    parcimonie: float,
    quantification: str,
    chemin_onnx: str,
) -> None:
    torch.manual_seed(0)
    chargeur_ent = cree_chargeur(
        f"{dossier_prepare}/entrainement", TAILLE_IMAGE, melange=True
    )
    images, etiquettes = rassemble_images(
        cree_chargeur(f"{dossier_prepare}/test", TAILLE_IMAGE)
    )
    modele = LeNet()
    modele.load_state_dict(torch.load(chemin_modele))
    with tempfile.TemporaryDirectory() as dossier_tmp:
        chemin_flottant = f"{dossier_tmp}/flottant.onnx"
        exporte_onnx(modele, chemin_flottant)
        if parcimonie > 0:
            types_couches = nn.Conv2d | nn.Linear
            poids = [
                (m, "weight") for m in modele.modules() if isinstance(m, types_couches)
            ]
            prune.global_unstructured(
                poids, pruning_method=prune.L1Unstructured, amount=parcimonie
            )
            optimiseur = torch.optim.AdamW(modele.parameters(), lr=taux_apprentissage)
            entraine_epoque(modele, chargeur_ent, optimiseur)
            for module, nom in poids:
                prune.remove(module, nom)
        chemin_elague = f"{dossier_tmp}/elague.onnx"
        exporte_onnx(modele, chemin_elague)
        if quantification == "aucune":
            shutil.copyfile(chemin_elague, chemin_onnx)
        else:
            chemin_pretraite = f"{dossier_tmp}/pretraite.onnx"
            quant_pre_process(chemin_elague, chemin_pretraite)
            if quantification == "dynamique":
                quantize_dynamic(
                    chemin_pretraite, chemin_onnx, weight_type=QuantType.QInt8
                )
            else:
                lot, _ = next(iter(chargeur_ent))
                quantize_static(
                    chemin_pretraite,
                    chemin_onnx,
                    Calibration([image[None] for image in lot]),
                    per_channel=True,
                )
        metriques = {
            "flottant": mesure(chemin_flottant, images, etiquettes),
            "compresse": mesure(chemin_onnx, images, etiquettes),
        }
    CHEMIN_METRIQUES.write_text(json.dumps(metriques, indent=2))
