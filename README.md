# Pipelines DVC et compression de modèles

*[English version](https://github.com/shuuchuu/compression/tree/main-en)*

Le projet des travaux pratiques *Pipelines DVC et compression de modèles*. Leur énoncé est ci-dessous, pour les faire dans un Codespace GitHub ou un clone local&nbsp;; c'est le même que celui du notebook des travaux pratiques, en [français](https://colab.research.google.com/github/shuuchuu/labs/blob/main/compression-hands-on-fr.ipynb) ou en [anglais](https://colab.research.google.com/github/shuuchuu/labs/blob/main/compression-hands-on-en.ipynb), qui les fait dans Colab.

Nous allons construire, avec DVC, la chaîne de traitements d'un classifieur d'images de paysages, en PyTorch&nbsp;: préparation des données, entraînement, puis compression du modèle, chaque étape mesurée et comparée d'une expérience à l'autre.

## Mise en place de l'environnement de travail

Ouvrez le dépôt dans un Codespace&nbsp;: sur GitHub, **Code** > **Codespaces** > **Create codespace on main**, ou [ce lien](https://codespaces.new/shuuchuu/compression). Dans un clone local, installez [uv](https://docs.astral.sh/uv/getting-started/installation/).

Les commandes ci-dessous se lancent dans un terminal à la racine du dépôt (**Terminal** > **Nouveau terminal** dans le Codespace). Les fichiers dont le code est donné sont à créer à l'endroit indiqué.

## Création du paquet Python et du dépôt DVC

Créez un paquet Python `compression_modele` avec `uv init`. Les roues de PyTorch publiées sur PyPI embarquent les bibliothèques CUDA, plusieurs gigaoctets, inutiles ici, où tout tourne sur CPU&nbsp;: avant d'installer quoi que ce soit, ajoutez à la fin de `pyproject.toml` l'index des roues CPU de PyTorch, qui ne sert (`explicit = true`) qu'aux paquets que `[tool.uv.sources]` lui attribue&nbsp;:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
torchvision = { index = "pytorch-cpu" }
```

Ajoutez ensuite au paquet, avec `uv add`, DVC et les bibliothèques qu'utilise le code fourni plus loin&nbsp;: `torch`, `torchvision`, `onnx`, `onnxscript` (dont l'export ONNX de PyTorch a besoin) et `onnxruntime`. Initialisez enfin DVC dans le dépôt avec `dvc init`, puis enregistrez le tout dans git. DVC se lance avec `uv run`, qui l'exécute dans l'environnement du projet.

**Documentation utile**

- [`uv init`](https://docs.astral.sh/uv/reference/cli/#uv-init)
- [`uv add`](https://docs.astral.sh/uv/reference/cli/#uv-add)
- [`dvc init`](https://doc.dvc.org/command-reference/init)

<details>
<summary>Conseils supplémentaires</summary>

`uv init --package --name compression_modele .` crée le paquet dans le dossier courant, ses sources dans `src/compression_modele/` (pas `compression`&nbsp;: la bibliothèque standard de Python&nbsp;3.14 a un paquet de ce nom, qui masquerait le nôtre). `uv add` prend plusieurs paquets à la fois, et crée l'environnement du projet, `.venv`, que git ignore. `dvc init` crée le dossier `.dvc/` et ajoute lui-même ses fichiers à l'index de git.

```console
uv init --package --name compression_modele .
```

À la fin de `pyproject.toml`&nbsp;:

```toml
...
```

```console
uv add dvc ...
uv run dvc init
git add .
git commit -m "..."
```

</details>

<details>
<summary>Solution</summary>

```console
uv init --package --name compression_modele .
```

À la fin de `pyproject.toml`&nbsp;:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
torchvision = { index = "pytorch-cpu" }
```

```console
uv add dvc onnx onnxruntime onnxscript torch torchvision
uv run dvc init
git add .
git commit -m "Création du paquet Python et du dépôt DVC"
```

</details>

## Ajout des données à DVC

Les commandes suivantes téléchargent dans `donnees/` le dossier `landscape` du dépôt [shuuchuu/datasets](https://github.com/shuuchuu/datasets/tree/main/landscape), sans `donnees/seg_pred/`, des images sans étiquette qui ne servent pas ici. `donnees/seg_train/` et `donnees/seg_test/` contiennent les images d'entraînement et de test (150&nbsp;×&nbsp;150&nbsp;pixels), un dossier par classe (`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`).

```console
curl -fsSL https://github.com/shuuchuu/datasets/raw/refs/heads/main/ghsparse.sh -o /tmp/ghsparse.sh && bash /tmp/ghsparse.sh landscape
mv landscape donnees
rm -r donnees/seg_pred
```

Ajoutez `donnees/` à DVC, puis enregistrez dans git les fichiers qui le décrivent.

<details>
<summary>Conseils supplémentaires</summary>

`dvc add` met le dossier dans le cache de DVC, `.dvc/cache`, et écrit `donnees.dvc`, qui retient son empreinte md5&nbsp;; il ajoute aussi `/donnees` à `.gitignore`&nbsp;: ce sont ces deux fichiers qui vont dans git, pas les images.

```console
uv run dvc add ...
git add ...
git commit -m "..."
```

</details>

<details>
<summary>Solution</summary>

```console
uv run dvc add donnees
git add .gitignore donnees.dvc
git commit -m "Ajout des données"
```

</details>

## Préparation des données

L'entraînement et la compression lisent les mêmes images. Plutôt que de les décoder et de les redimensionner à chaque fois, une première étape de la chaîne, `prepare`, le fait une fois pour toutes&nbsp;: elle copie les `images_par_classe` premières images de chaque classe dans `preparees/entrainement/` et `preparees/test/`, réduites à `TAILLE_IMAGE`, 88&nbsp;×&nbsp;88&nbsp;pixels. 500&nbsp;images par classe suffisent à ces travaux pratiques, et un entraînement sur CPU dure alors une à deux minutes. À cette taille, les dernières cartes de caractéristiques du LeNet de l'entraînement font 20&nbsp;×&nbsp;20, que sa couche `AdaptiveAvgPool2d` réduit à 5&nbsp;×&nbsp;5 par des moyennes sur des fenêtres régulières, ce que l'export ONNX traduit en une seule opération (à 100&nbsp;×&nbsp;100, il en faut une centaine).

Son code, dans `src/compression_modele/donnees.py`, transforme aussi les images en tenseurs, leurs pixels mis à l'échelle [0,&nbsp;1] (`cree_transformation`), et les charge par lots de 64 (`cree_chargeur`)&nbsp;:

```python
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
```

Le point d'entrée de la chaîne, `src/compression_modele/__main__.py`&nbsp;: `python -m compression_modele <étape>` lance la fonction de cette étape, `prepare` de `donnees.py`, puis `entraine` de `entrainement.py` et `compresse` de `compression.py`, que vous écrirez plus loin, avec les paramètres que `dvc.yaml` lui déclare, lus dans `params.yaml` par `dvc.api.params_show`. Le module d'une étape n'est importé que quand elle s'exécute&nbsp;: ce fichier ne changera plus.

```python
from importlib import import_module
from sys import argv

from dvc.api import params_show

MODULES = {"prepare": "donnees", "entraine": "entrainement", "compresse": "compression"}

if len(argv) != 2 or argv[1] not in MODULES:
    message = f"Usage : python -m compression_modele {{{','.join(MODULES)}}}"
    raise SystemExit(message)

etape = argv[1]
fonction = getattr(import_module(f"compression_modele.{MODULES[etape]}"), etape)
fonction(**params_show(stages=[etape]))
```

Ajoutez à la chaîne l'étape `prepare`&nbsp;:

1. Créez `params.yaml`, à la racine du dépôt, avec les paramètres de `prepare`&nbsp;: `dossier_ent` (`donnees/seg_train`), `dossier_test` (`donnees/seg_test`), `images_par_classe` (`500`) et `dossier_prepare` (`preparees`).
2. Avec `dvc stage add`, créez l'étape `prepare`, qui lance `python -m compression_modele prepare`, en déclarant ses paramètres, ses dépendances (les images et le code qu'elle exécute) et sa sortie.
3. Exécutez la chaîne avec `dvc repro`, puis enregistrez dans git le code, `params.yaml` et ce que DVC a écrit.

**Documentation utile**

- [`dvc stage add`](https://doc.dvc.org/command-reference/stage/add)
- [`dvc repro`](https://doc.dvc.org/command-reference/repro)
- [`dvc.api.params_show`](https://doc.dvc.org/api-reference/params_show)

<details>
<summary>Conseils supplémentaires</summary>

Les options de `dvc stage add`&nbsp;: `-n` nomme l'étape, `-p` déclare ses paramètres (leurs noms dans `params.yaml`, séparés par des virgules), `-d` chaque dépendance, `-o` sa sortie, puis vient la commande. Le code qu'exécute `prepare`, ce sont `__main__.py` et `donnees.py`. L'étape est écrite dans `dvc.yaml`. `dvc repro` lance les étapes dont une dépendance ou un paramètre a changé, et retient dans `dvc.lock` les empreintes de ce qu'elles ont lu et produit&nbsp;; DVC ajoute aussi `/preparees` à `.gitignore`.

`params.yaml`&nbsp;:

```yaml
dossier_ent: donnees/seg_train
dossier_test: ...
images_par_classe: ...
dossier_prepare: ...
```

```console
uv run dvc stage add -n prepare -p ... -d ... -o ... python -m compression_modele prepare
uv run dvc repro
git add ...
git commit -m "..."
```

</details>

<details>
<summary>Solution</summary>

`params.yaml`&nbsp;:

```yaml
dossier_ent: donnees/seg_train
dossier_test: donnees/seg_test
images_par_classe: 500
dossier_prepare: preparees
```

```console
uv run dvc stage add -n prepare \
    -p dossier_ent,dossier_test,images_par_classe,dossier_prepare \
    -d donnees/seg_train -d donnees/seg_test \
    -d src/compression_modele/__main__.py -d src/compression_modele/donnees.py \
    -o preparees \
    python -m compression_modele prepare
```

```console
uv run dvc repro
git add .gitignore dvc.lock dvc.yaml params.yaml src
git commit -m "Ajout de l'étape de préparation"
```

</details>

## Stockage distant

Le cache de DVC, `.dvc/cache`, n'existe que sur cette machine. Pour partager données et modèles, ou les retrouver ailleurs, DVC les envoie sur un stockage distant, déclaré une fois pour toutes dans `.dvc/config`. Ici, un simple dossier, `/tmp/stockage-dvc`, joue ce rôle.

1. Avec `dvc remote add`, déclarez le stockage `local`, dans `/tmp/stockage-dvc`, comme stockage par défaut, puis enregistrez la configuration dans git.
2. Envoyez-y les données avec `dvc push`, et regardez ce que contient `/tmp/stockage-dvc`.
3. Supprimez `preparees/`, puis récupérez-le avec `dvc pull`.

Que faudrait-il faire, sur une autre machine, pour retrouver `preparees/` sans relancer `prepare`&nbsp;?

**Documentation utile**

- [`dvc remote add`](https://doc.dvc.org/command-reference/remote/add)
- [`dvc push`](https://doc.dvc.org/command-reference/push)
- [`dvc pull`](https://doc.dvc.org/command-reference/pull)

<details>
<summary>Conseils supplémentaires</summary>

`-d` fait du stockage celui par défaut, qu'utilisent `dvc push` et `dvc pull` sans option. `dvc remote add` écrit dans `.dvc/config`, que git suit. Le stockage range chaque fichier sous son empreinte md5, comme le cache (`ls /tmp/stockage-dvc/files/md5`).

```console
uv run dvc remote add -d ... ...
git add ...
git commit -m "..."
uv run dvc push
rm -r preparees
uv run dvc pull
```

</details>

<details>
<summary>Solution</summary>

```console
uv run dvc remote add -d local /tmp/stockage-dvc
git add .dvc/config
git commit -m "Ajout du stockage distant"
uv run dvc push
ls /tmp/stockage-dvc/files/md5 | head
rm -r preparees
uv run dvc pull
ls preparees
```

`dvc push` envoie le contenu du cache que décrivent `donnees.dvc` et `dvc.lock`&nbsp;: les images de `donnees/` et celles de `preparees/`, chacune sous son empreinte. Ici, `dvc pull` ne télécharge rien, puisque le cache a encore les fichiers&nbsp;: il les remet à leur place (`dvc checkout`). Sur une autre machine, `git clone` puis `dvc pull` téléchargeraient `preparees/` depuis le stockage, sans rien recalculer, s'il était partagé (un compartiment S3, un serveur SSH…)&nbsp;: un dossier de `/tmp` ne l'est pas.

</details>

## Entraînement d'un modèle simple

Nous allons maintenant entraîner un modèle de classification d'images&nbsp;: le modèle historique LeNet, deux convolutions puis trois couches denses. Voici son code, à mettre dans `src/compression_modele/modele.py`&nbsp;:

```python
import torch
from torch import nn


class LeNet(nn.Module):
    def __init__(self, nb_classes: int = 6) -> None:
        super().__init__()
        self.caracteristiques = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((5, 5)),
        )
        self.classifieur = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, nb_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifieur(self.caracteristiques(x))
```

Et celui de l'entraînement, dans `src/compression_modele/entrainement.py`&nbsp;: `entraine_epoque` passe une fois sur les images d'entraînement, `evalue` mesure l'exactitude (*accuracy*, la part d'images bien classées), et `entraine` enchaîne `nb_epoques` époques, puis enregistre les poids du modèle dans `chemin_modele` et son exactitude sur les images de test dans `metriques/entrainement.json`, un fichier de métriques que DVC sait lire et comparer&nbsp;:

```python
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
```

Ajoutez à la chaîne l'étape `entraine`, qui lance `python -m compression_modele entraine`&nbsp;: ses paramètres `taux_apprentissage` (`0.001`), `nb_epoques` (`5`) et `chemin_modele` (`classifieur_paysages.pt`) s'ajoutent à `params.yaml`, et elle lit aussi `dossier_prepare`. Déclarez `metriques/entrainement.json` comme métrique de l'étape, exécutez la chaîne, affichez les métriques avec `dvc metrics show`, puis enregistrez le tout dans git.

**Documentation utile**

- [`dvc metrics show`](https://doc.dvc.org/command-reference/metrics/show)

<details>
<summary>Conseils supplémentaires</summary>

Les dépendances de `entraine` sont la sortie de `prepare` et le code qu'elle exécute (`__main__.py`, `donnees.py`, `modele.py` et `entrainement.py`)&nbsp;: c'est en déclarant `preparees` comme dépendance que DVC sait que `entraine` vient après `prepare`. Sa sortie est le modèle (`-o`). Sa métrique se déclare avec `-M`&nbsp;: un fichier que DVC ne met pas en cache et que git suit, pour comparer les métriques d'un commit à l'autre. `dvc repro` ne relance pas `prepare`, qui est à jour. L'entraînement dure une à deux minutes.

À la fin de `params.yaml`&nbsp;:

```yaml
taux_apprentissage: ...
nb_epoques: ...
chemin_modele: ...
```

```console
uv run dvc stage add -n entraine -p ... -d preparees -d ... -o ... -M ... python -m compression_modele entraine
uv run dvc repro
uv run dvc metrics show
git add ...
git commit -m "..."
```

</details>

<details>
<summary>Solution</summary>

À la fin de `params.yaml`&nbsp;:

```yaml
taux_apprentissage: 0.001
nb_epoques: 5
chemin_modele: classifieur_paysages.pt
```

```console
uv run dvc stage add -n entraine \
    -p dossier_prepare,taux_apprentissage,nb_epoques,chemin_modele \
    -d preparees -d src/compression_modele/__main__.py -d src/compression_modele/donnees.py \
    -d src/compression_modele/modele.py -d src/compression_modele/entrainement.py \
    -o classifieur_paysages.pt -M metriques/entrainement.json \
    python -m compression_modele entraine
```

```console
uv run dvc repro
uv run dvc metrics show
git add .gitignore dvc.lock dvc.yaml metriques params.yaml src
git commit -m "Ajout de l'étape d'entraînement"
```

</details>

### Une expérience&nbsp;: changement du taux d'apprentissage

Lancez avec `dvc exp run` une expérience où `taux_apprentissage` vaut `0.003`, sans modifier `params.yaml` à la main. Comparez ses métriques à celles du dernier commit avec `dvc metrics diff`, puis listez les expériences avec `dvc exp show`. Quelles étapes l'expérience a-t-elle exécutées&nbsp;? Lesquelles s'exécuteraient si vous changiez `images_par_classe`&nbsp;? Si l'exactitude s'améliore, gardez ce taux&nbsp;: enregistrez dans git ce que l'expérience a changé.

**Documentation utile**

- [`dvc exp run`](https://doc.dvc.org/command-reference/exp/run)
- [`dvc metrics diff`](https://doc.dvc.org/command-reference/metrics/diff)
- [`dvc exp show`](https://doc.dvc.org/command-reference/exp/show)

<details>
<summary>Conseils supplémentaires</summary>

`-S taux_apprentissage=0.003` change le paramètre le temps de l'expérience, et `dvc exp run` en laisse le résultat dans l'espace de travail&nbsp;: `params.yaml`, `dvc.lock`, le modèle et `metriques/entrainement.json`. `dvc metrics diff` compare l'espace de travail au dernier commit&nbsp;; `--only-changed` n'affiche, dans `dvc exp show`, que les colonnes qui diffèrent d'une expérience à l'autre. Pour `images_par_classe`, regardez de quoi dépend `entraine`.

```console
uv run dvc exp run -S ...
uv run dvc metrics diff
uv run dvc exp show --only-changed
git add ...
git commit -m "..."
```

</details>

<details>
<summary>Solution</summary>

```console
uv run dvc exp run -S taux_apprentissage=0.003
uv run dvc metrics diff
uv run dvc exp show --only-changed
git add dvc.lock metriques params.yaml
git commit -m "Taux d'apprentissage à 0,003"
```

Seule `entraine` s'exécute&nbsp;: `prepare` ne lit pas `taux_apprentissage`. Sur nos essais, l'exactitude passe de 0,58 à 0,61 environ (elle varie un peu d'une machine à l'autre). Changer `images_par_classe` rendrait `prepare` obsolète, et `entraine` s'exécuterait après elle, parce que sa dépendance `preparees` aurait changé. Si `prepare` produisait exactement les mêmes fichiers, DVC ne relancerait pas `entraine`&nbsp;: il compare les empreintes, pas les dates. `dvc exp show` garde la trace de chaque expérience, même non commitée&nbsp;: `dvc exp apply` en ramène une dans l'espace de travail, `dvc exp branch` en fait une branche.

</details>

## Compression du modèle a posteriori

Reste l'étape `compresse`, qui réduit le modèle entraîné et mesure ce que la réduction coûte et rapporte, selon deux paramètres&nbsp;:

- `parcimonie`&nbsp;: la part des poids des convolutions et des couches denses à mettre à zéro, par un élagage non structuré, global, des poids de plus faible magnitude, suivi d'une époque de réentraînement&nbsp;; `0` n'élague rien&nbsp;;
- `quantification`&nbsp;: `aucune` (pas de quantification), `dynamique` ou `statique`, les quantifications post-entraînement en int8 d'ONNX Runtime.

Complétez la fonction `compresse` de `src/compression_modele/compression.py`, dont le début fournit&nbsp;:

- `exporte_onnx`, qui exporte un modèle au format ONNX, en un seul fichier, son entrée nommée `x`&nbsp;;
- `Calibration`, qui donne à `quantize_static` des images une à une&nbsp;;
- `mesure`, qui mesure l'exactitude d'un modèle ONNX sur des images, la taille de son fichier, brute (`taille_mo`) et compressée par gzip (`gzip_mo`), et sa latence médiane pour une image (`latence_ms`)&nbsp;;
- `rassemble_images`, qui rassemble les images et les étiquettes d'un `DataLoader` en tableaux.

La fonction doit&nbsp;:

1. charger le LeNet entraîné (`chemin_modele`) et l'exporter en ONNX&nbsp;: c'est le modèle flottant de référence&nbsp;;
2. si `parcimonie` est positif, élaguer le modèle, le réentraîner une époque avec `entraine_epoque` et un optimiseur `AdamW` de taux `taux_apprentissage`, puis rendre l'élagage définitif&nbsp;;
3. exporter le modèle élagué, puis le quantifier selon `quantification` dans `chemin_onnx` (`aucune`&nbsp;: une simple copie), en calibrant la quantification statique sur un lot d'images d'entraînement&nbsp;;
4. écrire dans `CHEMIN_METRIQUES`, en JSON, les mesures du modèle flottant et du modèle réduit sur les images de test, sous les clés `flottant` et `compresse`.

**Documentation utile**

- [`torch.load`](https://docs.pytorch.org/docs/stable/generated/torch.load.html)
- [`torch.nn.Module.load_state_dict`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict)
- [`tempfile.TemporaryDirectory`](https://docs.python.org/3/library/tempfile.html#tempfile.TemporaryDirectory)
- [`shutil.copyfile`](https://docs.python.org/3/library/shutil.html#shutil.copyfile)

```python
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

    def get_next(self) -> dict | None:
        image = next(self.images, None)
        return None if image is None else {"x": image.numpy()}


def mesure(chemin: str, images: np.ndarray, etiquettes: np.ndarray) -> dict:
    session = ort.InferenceSession(chemin, providers=["CPUExecutionProvider"])
    predictions, durees = [], []
    for image in images:
        debut = time.perf_counter()
        (logits,) = session.run(None, {"x": image[None]})
        durees.append(time.perf_counter() - debut)
        predictions.append(logits.argmax())
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
    # Votre code ici
    ...
```

<details>
<summary>Conseils supplémentaires</summary>

Les exports et les quantifications écrivent des fichiers intermédiaires (le modèle flottant, le modèle élagué, celui que prépare `quant_pre_process`)&nbsp;: un dossier temporaire les garde, le temps de la fonction. L'élagage est celui des diapositives, avec `amount=parcimonie`. `quantize_dynamic` et `quantize_static` lisent le modèle préparé par `quant_pre_process`, celui-ci avec `Calibration([image[None] for image in lot])` et `per_channel=True`, `lot` étant le premier lot de `chargeur_ent`. `rassemble_images(chargeur_test)` renvoie les images et les étiquettes que `mesure` attend.

```python
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
    modele.load_state_dict(...)
    with tempfile.TemporaryDirectory() as dossier_tmp:
        chemin_flottant = f"{dossier_tmp}/flottant.onnx"
        exporte_onnx(modele, chemin_flottant)
        if parcimonie > 0:
            ...  # élaguer, réentraîner une époque, rendre l'élagage définitif
        chemin_elague = f"{dossier_tmp}/elague.onnx"
        exporte_onnx(modele, chemin_elague)
        if quantification == "aucune":
            shutil.copyfile(...)
        else:
            chemin_pretraite = f"{dossier_tmp}/pretraite.onnx"
            quant_pre_process(...)
            if quantification == "dynamique":
                quantize_dynamic(...)
            else:
                lot, _ = next(iter(chargeur_ent))
                quantize_static(...)
        metriques = {"flottant": mesure(...), "compresse": mesure(...)}
    CHEMIN_METRIQUES.write_text(json.dumps(metriques, indent=2))
```

</details>

<details>
<summary>Solution</summary>

`src/compression_modele/compression.py`&nbsp;:

```python
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

    def get_next(self) -> dict | None:
        image = next(self.images, None)
        return None if image is None else {"x": image.numpy()}


def mesure(chemin: str, images: np.ndarray, etiquettes: np.ndarray) -> dict:
    session = ort.InferenceSession(chemin, providers=["CPUExecutionProvider"])
    predictions, durees = [], []
    for image in images:
        debut = time.perf_counter()
        (logits,) = session.run(None, {"x": image[None]})
        durees.append(time.perf_counter() - debut)
        predictions.append(logits.argmax())
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
```

</details>

## L'étape de compression

Ajoutez à `params.yaml` `parcimonie` (`0.5`), `quantification` (`statique`) et `chemin_onnx` (`classifieur_paysages.onnx`), et à la chaîne l'étape `compresse`, qui lance `python -m compression_modele compresse`, avec ses paramètres, ses dépendances, sa sortie et sa métrique, `metriques/compression.json`. Exécutez la chaîne, comparez avec `dvc metrics show` le modèle flottant et le modèle réduit, puis enregistrez le tout dans git. Le modèle réduit est-il plus petit&nbsp;? Plus rapide&nbsp;? Aussi exact&nbsp;?

<details>
<summary>Conseils supplémentaires</summary>

Les paramètres de `compresse` sont les trois nouveaux, plus `dossier_prepare`, `chemin_modele` et `taux_apprentissage`, que la fonction lit aussi. Ses dépendances sont `preparees`, le modèle entraîné et le code qu'elle exécute&nbsp;: `__main__.py`, `compression.py`, et les modules qu'il importe.

À la fin de `params.yaml`&nbsp;:

```yaml
parcimonie: ...
quantification: ...
chemin_onnx: ...
```

```console
uv run dvc stage add -n compresse -p ... -d ... -o ... -M ... python -m compression_modele compresse
uv run dvc repro
uv run dvc metrics show
git add ...
git commit -m "..."
```

</details>

<details>
<summary>Solution</summary>

À la fin de `params.yaml`&nbsp;:

```yaml
parcimonie: 0.5
quantification: statique
chemin_onnx: classifieur_paysages.onnx
```

```console
uv run dvc stage add -n compresse \
    -p dossier_prepare,chemin_modele,taux_apprentissage,parcimonie,quantification,chemin_onnx \
    -d preparees -d classifieur_paysages.pt -d src/compression_modele/__main__.py \
    -d src/compression_modele/compression.py -d src/compression_modele/donnees.py \
    -d src/compression_modele/modele.py -d src/compression_modele/entrainement.py \
    -o classifieur_paysages.onnx -M metriques/compression.json \
    python -m compression_modele compresse
```

```console
uv run dvc repro
uv run dvc metrics show
git add .gitignore dvc.lock dvc.yaml metriques params.yaml src
git commit -m "Ajout de l'étape de compression"
```

Sur nos essais, le modèle élagué de moitié puis quantifié en int8 est trois fois plus petit (0,27 → 0,09&nbsp;Mo), à une exactitude d'un point plus basse (0,615 → 0,605)… mais plus lent&nbsp;: 0,7&nbsp;ms par image au lieu de 0,45. Ce surcoût ne vient pas des conversions du format QDQ&nbsp;: à l'ouverture de la session, ONNX Runtime fond le graphe quantifié en opérateurs int8 (`QLinearConv` pour les convolutions, `QGemm` pour les couches linéaires), avec une seule quantification à l'entrée et une seule déquantification à la sortie. Sur la seule machine où nous l'avons mesuré, il vient des noyaux int8 eux-mêmes, lents sur un CPU sans instructions int8 rapides (VNNI), et des changements de disposition des données qu'ils demandent, que rien ne compense sur un modèle aussi petit. Un gain annoncé n'est pas un gain mesuré&nbsp;: mesurez sur le matériel cible.

</details>

### Expériences de compression

Lancez deux expériences&nbsp;: une quantification dynamique (`quantification` à `dynamique`), et un élagage de 90&nbsp;% sans quantification (`parcimonie` à `0.9`, `quantification` à `aucune`). Comparez-les avec `dvc exp show`. Quel réglage donne le plus petit fichier&nbsp;? Le modèle le plus rapide&nbsp;? L'élagage seul réduit-il la taille du fichier&nbsp;? Et celle du fichier compressé&nbsp;?

<details>
<summary>Conseils supplémentaires</summary>

`-S` se répète pour changer plusieurs paramètres. Seule `compresse` s'exécute.

```console
uv run dvc exp run -S ...
uv run dvc exp run -S ... -S ...
uv run dvc exp show --only-changed
```

</details>

<details>
<summary>Solution</summary>

```console
uv run dvc exp run -S quantification=dynamique
uv run dvc exp run -S parcimonie=0.9 -S quantification=aucune
uv run dvc exp show --only-changed
```

Les deux quantifications donnent des fichiers trois fois plus petits, et la dynamique est la plus lente (2,5&nbsp;ms par image sur nos essais)&nbsp;: elle calcule les plages des activations à chaque inférence, et les convolutions en entiers d'ONNX Runtime (`ConvInteger`) sont lentes sur CPU. L'élagage seul ne réduit pas la taille du fichier&nbsp;: les zéros y sont stockés comme les autres poids, et les calculs ne vont pas plus vite. Mais le fichier compressé est cinq fois plus petit (0,23 → 0,05&nbsp;Mo)&nbsp;: les zéros se compressent bien, ce qui compte pour télécharger un modèle. Et l'exactitude ne baisse pas&nbsp;: l'époque de réentraînement compense l'élagage, signe que ce LeNet est surdimensionné pour ces images.

</details>

## Solution

Le projet terminé est sur la [branche `solution`](https://github.com/shuuchuu/compression/tree/solution) du dépôt.
