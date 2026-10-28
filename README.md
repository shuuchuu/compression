# Pipelines DVC et compression de modèles

*[English version](https://github.com/shuuchuu/compression/tree/main-en)*

Le projet des travaux pratiques *Pipelines DVC et compression de modèles*. Leur énoncé est ci-dessous, pour les faire dans un Codespace GitHub ou un clone local&nbsp;; c'est le même que celui du notebook des travaux pratiques, en [français](https://colab.research.google.com/github/shuuchuu/labs/blob/main/compression-hands-on-fr.ipynb) ou en [anglais](https://colab.research.google.com/github/shuuchuu/labs/blob/main/compression-hands-on-en.ipynb), qui les fait dans Colab.

Nous allons construire, avec DVC, la chaîne de traitements d'un classifieur d'images de paysages, en PyTorch&nbsp;: préparation des données, entraînement, puis compression du modèle, chaque étape mesurée et comparée d'une expérience à l'autre.

## Mise en place de l'environnement de travail

Ouvrez le dépôt dans un Codespace&nbsp;: sur GitHub, **Code** > **Codespaces** > **Create codespace on main**, ou [ce lien](https://codespaces.new/shuuchuu/compression). Dans un clone local, installez [uv](https://docs.astral.sh/uv/getting-started/installation/).

Les commandes ci-dessous se lancent dans un terminal à la racine du dépôt (**Terminal** > **Nouveau terminal** dans le Codespace). Les fichiers dont le code est donné sont à créer à l'endroit indiqué.

## Création du paquet Python et du dépôt DVC

Créez un paquet Python `model_compression` avec `uv init`. Les roues de PyTorch publiées sur PyPI embarquent les bibliothèques CUDA, plusieurs gigaoctets, inutiles ici, où tout tourne sur CPU&nbsp;: avant d'installer quoi que ce soit, ajoutez à la fin de `pyproject.toml` l'index des roues CPU de PyTorch, qui ne sert (`explicit = true`) qu'aux paquets que `[tool.uv.sources]` lui attribue&nbsp;:

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

`uv init --package --name model_compression .` crée le paquet dans le dossier courant, ses sources dans `src/model_compression/` (pas `compression`&nbsp;: la bibliothèque standard de Python&nbsp;3.14 a un paquet de ce nom, qui masquerait le nôtre). `uv add` prend plusieurs paquets à la fois, et crée l'environnement du projet, `.venv`, que git ignore. `dvc init` crée le dossier `.dvc/` et ajoute lui-même ses fichiers à l'index de git.

```console
uv init --package --name model_compression .
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
uv init --package --name model_compression .
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

Les commandes suivantes téléchargent dans `data/` le dossier `landscape` du dépôt [shuuchuu/datasets](https://github.com/shuuchuu/datasets/tree/main/landscape), sans `data/seg_pred/`, des images sans étiquette qui ne servent pas ici. `data/seg_train/` et `data/seg_test/` contiennent les images d'entraînement et de test (150&nbsp;×&nbsp;150&nbsp;pixels), un dossier par classe (`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`).

```console
curl -fsSL https://github.com/shuuchuu/datasets/raw/refs/heads/main/ghsparse.sh -o /tmp/ghsparse.sh && bash /tmp/ghsparse.sh landscape
mv landscape data
rm -r data/seg_pred
```

Ajoutez `data/` à DVC, puis enregistrez dans git les fichiers qui le décrivent.

<details>
<summary>Conseils supplémentaires</summary>

`dvc add` met le dossier dans le cache de DVC, `.dvc/cache`, et écrit `data.dvc`, qui retient son empreinte md5&nbsp;; il ajoute aussi `/data` à `.gitignore`&nbsp;: ce sont ces deux fichiers qui vont dans git, pas les images.

```console
uv run dvc add ...
git add ...
git commit -m "..."
```

</details>

<details>
<summary>Solution</summary>

```console
uv run dvc add data
git add .gitignore data.dvc
git commit -m "Ajout des données"
```

</details>

## Préparation des données

L'entraînement et la compression lisent les mêmes images. Plutôt que de les décoder et de les redimensionner à chaque fois, une première étape de la chaîne, `prepare`, le fait une fois pour toutes&nbsp;: elle copie les `images_per_class` premières images de chaque classe dans `prepared/train/` et `prepared/test/`, réduites à `IMAGE_SIZE`, 88&nbsp;×&nbsp;88&nbsp;pixels. 500&nbsp;images par classe suffisent à ces travaux pratiques, et un entraînement sur CPU dure alors une à deux minutes. À cette taille, les dernières cartes de caractéristiques du LeNet de l'entraînement font 20&nbsp;×&nbsp;20, que sa couche `AdaptiveAvgPool2d` réduit à 5&nbsp;×&nbsp;5 par des moyennes sur des fenêtres régulières, ce que l'export ONNX traduit en une seule opération (à 100&nbsp;×&nbsp;100, il en faut une centaine).

Son code, dans `src/model_compression/data.py`, transforme aussi les images en tenseurs, leurs pixels mis à l'échelle [0,&nbsp;1] (`get_transform`), et les charge par lots de 64 (`get_loader`)&nbsp;:

```python
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

IMAGE_SIZE = (88, 88)


def get_transform(image_size: tuple[int, int]) -> v2.Compose:
    return v2.Compose(
        [v2.ToImage(), v2.Resize(image_size), v2.ToDtype(torch.float32, scale=True)]
    )


def get_loader(
    directory: str, image_size: tuple[int, int], shuffle: bool = False
) -> DataLoader:
    dataset = datasets.ImageFolder(directory, transform=get_transform(image_size))
    return DataLoader(dataset, batch_size=64, shuffle=shuffle)


def prepare(
    train_data: str, test_data: str, images_per_class: int, prepared_data: str
) -> None:
    for split, directory in (("train", train_data), ("test", test_data)):
        for class_dir in sorted(Path(directory).iterdir()):
            output_dir = Path(prepared_data) / split / class_dir.name
            output_dir.mkdir(parents=True, exist_ok=True)
            for path in sorted(class_dir.iterdir())[:images_per_class]:
                image = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
                image.save(output_dir / path.name, quality=95)
```

Le point d'entrée de la chaîne, `src/model_compression/__main__.py`&nbsp;: `python -m model_compression <étape>` lance la fonction de cette étape, `prepare` de `data.py`, puis `train` de `train.py` et `compress` de `compress.py`, que vous écrirez plus loin, avec les paramètres que `dvc.yaml` lui déclare, lus dans `params.yaml` par `dvc.api.params_show`. Le module d'une étape n'est importé que quand elle s'exécute&nbsp;: ce fichier ne changera plus.

```python
from importlib import import_module
from sys import argv

from dvc.api import params_show

MODULES = {"prepare": "data", "train": "train", "compress": "compress"}

if len(argv) != 2 or argv[1] not in MODULES:
    msg = f"Usage: python -m model_compression {{{','.join(MODULES)}}}"
    raise SystemExit(msg)

stage = argv[1]
function = getattr(import_module(f"model_compression.{MODULES[stage]}"), stage)
function(**params_show(stages=[stage]))
```

Ajoutez à la chaîne l'étape `prepare`&nbsp;:

1. Créez `params.yaml`, à la racine du dépôt, avec les paramètres de `prepare`&nbsp;: `train_data` (`data/seg_train`), `test_data` (`data/seg_test`), `images_per_class` (`500`) et `prepared_data` (`prepared`).
2. Avec `dvc stage add`, créez l'étape `prepare`, qui lance `python -m model_compression prepare`, en déclarant ses paramètres, ses dépendances (les images et le code qu'elle exécute) et sa sortie.
3. Exécutez la chaîne avec `dvc repro`, puis enregistrez dans git le code, `params.yaml` et ce que DVC a écrit.

**Documentation utile**

- [`dvc stage add`](https://doc.dvc.org/command-reference/stage/add)
- [`dvc repro`](https://doc.dvc.org/command-reference/repro)
- [`dvc.api.params_show`](https://doc.dvc.org/api-reference/params_show)

<details>
<summary>Conseils supplémentaires</summary>

Les options de `dvc stage add`&nbsp;: `-n` nomme l'étape, `-p` déclare ses paramètres (leurs noms dans `params.yaml`, séparés par des virgules), `-d` chaque dépendance, `-o` sa sortie, puis vient la commande. Le code qu'exécute `prepare`, ce sont `__main__.py` et `data.py`. L'étape est écrite dans `dvc.yaml`. `dvc repro` lance les étapes dont une dépendance ou un paramètre a changé, et retient dans `dvc.lock` les empreintes de ce qu'elles ont lu et produit&nbsp;; DVC ajoute aussi `/prepared` à `.gitignore`.

`params.yaml`&nbsp;:

```yaml
train_data: data/seg_train
test_data: ...
images_per_class: ...
prepared_data: ...
```

```console
uv run dvc stage add -n prepare -p ... -d ... -o ... python -m model_compression prepare
uv run dvc repro
git add ...
git commit -m "..."
```

</details>

<details>
<summary>Solution</summary>

`params.yaml`&nbsp;:

```yaml
train_data: data/seg_train
test_data: data/seg_test
images_per_class: 500
prepared_data: prepared
```

```console
uv run dvc stage add -n prepare \
    -p train_data,test_data,images_per_class,prepared_data \
    -d data/seg_train -d data/seg_test \
    -d src/model_compression/__main__.py -d src/model_compression/data.py \
    -o prepared \
    python -m model_compression prepare
```

```console
uv run dvc repro
git add .gitignore dvc.lock dvc.yaml params.yaml src
git commit -m "Ajout de l'étape de préparation"
```

</details>

## Stockage distant

Le cache de DVC, `.dvc/cache`, n'existe que sur cette machine. Pour partager données et modèles, ou les retrouver ailleurs, DVC les envoie sur un stockage distant, déclaré une fois pour toutes dans `.dvc/config`. Ici, un simple dossier, `/tmp/dvc-store`, joue ce rôle.

1. Avec `dvc remote add`, déclarez le stockage `local`, dans `/tmp/dvc-store`, comme stockage par défaut, puis enregistrez la configuration dans git.
2. Envoyez-y les données avec `dvc push`, et regardez ce que contient `/tmp/dvc-store`.
3. Supprimez `prepared/`, puis récupérez-le avec `dvc pull`.

Que faudrait-il faire, sur une autre machine, pour retrouver `prepared/` sans relancer `prepare`&nbsp;?

**Documentation utile**

- [`dvc remote add`](https://doc.dvc.org/command-reference/remote/add)
- [`dvc push`](https://doc.dvc.org/command-reference/push)
- [`dvc pull`](https://doc.dvc.org/command-reference/pull)

<details>
<summary>Conseils supplémentaires</summary>

`-d` fait du stockage celui par défaut, qu'utilisent `dvc push` et `dvc pull` sans option. `dvc remote add` écrit dans `.dvc/config`, que git suit. Le stockage range chaque fichier sous son empreinte md5, comme le cache (`ls /tmp/dvc-store/files/md5`).

```console
uv run dvc remote add -d ... ...
git add ...
git commit -m "..."
uv run dvc push
rm -r prepared
uv run dvc pull
```

</details>

<details>
<summary>Solution</summary>

```console
uv run dvc remote add -d local /tmp/dvc-store
git add .dvc/config
git commit -m "Ajout du stockage distant"
uv run dvc push
ls /tmp/dvc-store/files/md5 | head
rm -r prepared
uv run dvc pull
ls prepared
```

`dvc push` envoie le contenu du cache que décrivent `data.dvc` et `dvc.lock`&nbsp;: les images de `data/` et celles de `prepared/`, chacune sous son empreinte. Ici, `dvc pull` ne télécharge rien, puisque le cache a encore les fichiers&nbsp;: il les remet à leur place (`dvc checkout`). Sur une autre machine, `git clone` puis `dvc pull` téléchargeraient `prepared/` depuis le stockage, sans rien recalculer, s'il était partagé (un compartiment S3, un serveur SSH…)&nbsp;: un dossier de `/tmp` ne l'est pas.

</details>

## Entraînement d'un modèle simple

Nous allons maintenant entraîner un modèle de classification d'images&nbsp;: le modèle historique LeNet, deux convolutions puis trois couches denses. Voici son code, à mettre dans `src/model_compression/model.py`&nbsp;:

```python
import torch
from torch import nn


class LeNet(nn.Module):
    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((5, 5)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
```

Et celui de l'entraînement, dans `src/model_compression/train.py`&nbsp;: `train_epoch` passe une fois sur les images d'entraînement, `evaluate` mesure l'exactitude (*accuracy*, la part d'images bien classées), et `train` enchaîne `epochs` époques, puis enregistre les poids du modèle dans `model_path` et son exactitude sur les images de test dans `metrics/train.json`, un fichier de métriques que DVC sait lire et comparer&nbsp;:

```python
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import IMAGE_SIZE, get_loader
from .model import LeNet

METRICS_PATH = Path("metrics/train.json")


def train_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer
) -> float:
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    total, count = 0.0, 0
    for images, labels in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(labels)
        count += len(labels)
    return total / count


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, count = 0, 0
    with torch.inference_mode():
        for images, labels in loader:
            correct += int((model(images).argmax(dim=1) == labels).sum())
            count += len(labels)
    return correct / count


def train(
    prepared_data: str, learning_rate: float, epochs: int, model_path: str
) -> None:
    torch.manual_seed(0)
    train_loader = get_loader(f"{prepared_data}/train", IMAGE_SIZE, shuffle=True)
    test_loader = get_loader(f"{prepared_data}/test", IMAGE_SIZE)
    model = LeNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer)
        print(f"epoch {epoch + 1}/{epochs}: loss={loss:.3f}")
    accuracy = evaluate(model, test_loader)
    print(f"accuracy={accuracy:.3f}")
    torch.save(model.state_dict(), model_path)
    METRICS_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.write_text(json.dumps({"accuracy": accuracy, "loss": loss}))
```

Ajoutez à la chaîne l'étape `train`, qui lance `python -m model_compression train`&nbsp;: ses paramètres `learning_rate` (`0.001`), `epochs` (`5`) et `model_path` (`landscape_classifier.pt`) s'ajoutent à `params.yaml`, et elle lit aussi `prepared_data`. Déclarez `metrics/train.json` comme métrique de l'étape, exécutez la chaîne, affichez les métriques avec `dvc metrics show`, puis enregistrez le tout dans git.

**Documentation utile**

- [`dvc metrics show`](https://doc.dvc.org/command-reference/metrics/show)

<details>
<summary>Conseils supplémentaires</summary>

Les dépendances de `train` sont la sortie de `prepare` et le code qu'elle exécute (`__main__.py`, `data.py`, `model.py` et `train.py`)&nbsp;: c'est en déclarant `prepared` comme dépendance que DVC sait que `train` vient après `prepare`. Sa sortie est le modèle (`-o`). Sa métrique se déclare avec `-M`&nbsp;: un fichier que DVC ne met pas en cache et que git suit, pour comparer les métriques d'un commit à l'autre. `dvc repro` ne relance pas `prepare`, qui est à jour. L'entraînement dure une à deux minutes.

À la fin de `params.yaml`&nbsp;:

```yaml
learning_rate: ...
epochs: ...
model_path: ...
```

```console
uv run dvc stage add -n train -p ... -d prepared -d ... -o ... -M ... python -m model_compression train
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
learning_rate: 0.001
epochs: 5
model_path: landscape_classifier.pt
```

```console
uv run dvc stage add -n train \
    -p prepared_data,learning_rate,epochs,model_path \
    -d prepared -d src/model_compression/__main__.py -d src/model_compression/data.py \
    -d src/model_compression/model.py -d src/model_compression/train.py \
    -o landscape_classifier.pt -M metrics/train.json \
    python -m model_compression train
```

```console
uv run dvc repro
uv run dvc metrics show
git add .gitignore dvc.lock dvc.yaml metrics params.yaml src
git commit -m "Ajout de l'étape d'entraînement"
```

</details>

### Une expérience&nbsp;: changement du taux d'apprentissage

Lancez avec `dvc exp run` une expérience où `learning_rate` vaut `0.003`, sans modifier `params.yaml` à la main. Comparez ses métriques à celles du dernier commit avec `dvc metrics diff`, puis listez les expériences avec `dvc exp show`. Quelles étapes l'expérience a-t-elle exécutées&nbsp;? Lesquelles s'exécuteraient si vous changiez `images_per_class`&nbsp;? Si l'exactitude s'améliore, gardez ce taux&nbsp;: enregistrez dans git ce que l'expérience a changé.

**Documentation utile**

- [`dvc exp run`](https://doc.dvc.org/command-reference/exp/run)
- [`dvc metrics diff`](https://doc.dvc.org/command-reference/metrics/diff)
- [`dvc exp show`](https://doc.dvc.org/command-reference/exp/show)

<details>
<summary>Conseils supplémentaires</summary>

`-S learning_rate=0.003` change le paramètre le temps de l'expérience, et `dvc exp run` en laisse le résultat dans l'espace de travail&nbsp;: `params.yaml`, `dvc.lock`, le modèle et `metrics/train.json`. `dvc metrics diff` compare l'espace de travail au dernier commit&nbsp;; `--only-changed` n'affiche, dans `dvc exp show`, que les colonnes qui diffèrent d'une expérience à l'autre. Pour `images_per_class`, regardez de quoi dépend `train`.

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
uv run dvc exp run -S learning_rate=0.003
uv run dvc metrics diff
uv run dvc exp show --only-changed
git add dvc.lock metrics params.yaml
git commit -m "Taux d'apprentissage à 0,003"
```

Seule `train` s'exécute&nbsp;: `prepare` ne lit pas `learning_rate`. Sur nos essais, l'exactitude passe de 0,58 à 0,61 environ (elle varie un peu d'une machine à l'autre). Changer `images_per_class` rendrait `prepare` obsolète, et `train` s'exécuterait après elle, parce que sa dépendance `prepared` aurait changé. Si `prepare` produisait exactement les mêmes fichiers, DVC ne relancerait pas `train`&nbsp;: il compare les empreintes, pas les dates. `dvc exp show` garde la trace de chaque expérience, même non commitée&nbsp;: `dvc exp apply` en ramène une dans l'espace de travail, `dvc exp branch` en fait une branche.

</details>

## Compression du modèle a posteriori

Reste l'étape `compress`, qui réduit le modèle entraîné et mesure ce que la réduction coûte et rapporte, selon deux paramètres&nbsp;:

- `sparsity`&nbsp;: la part des poids des convolutions et des couches denses à mettre à zéro, par un élagage non structuré, global, des poids de plus faible magnitude, suivi d'une époque de réentraînement&nbsp;; `0` n'élague rien&nbsp;;
- `quant_type`&nbsp;: `none` (pas de quantification), `dynamic` ou `static`, les quantifications post-entraînement en int8 d'ONNX Runtime.

Complétez la fonction `compress` de `src/model_compression/compress.py`, dont le début fournit&nbsp;:

- `export_onnx`, qui exporte un modèle au format ONNX, en un seul fichier, son entrée nommée `x`&nbsp;;
- `Calibration`, qui donne à `quantize_static` des images une à une&nbsp;;
- `measure`, qui mesure l'exactitude d'un modèle ONNX sur des images, la taille de son fichier, brute (`size_mb`) et compressée par gzip (`gzip_mb`), et sa latence médiane pour une image (`latency_ms`)&nbsp;;
- `load_images`, qui rassemble les images et les étiquettes d'un `DataLoader` en tableaux.

La fonction doit&nbsp;:

1. charger le LeNet entraîné (`model_path`) et l'exporter en ONNX&nbsp;: c'est le modèle flottant de référence&nbsp;;
2. si `sparsity` est positif, élaguer le modèle, le réentraîner une époque avec `train_epoch` et un optimiseur `AdamW` de taux `learning_rate`, puis rendre l'élagage définitif&nbsp;;
3. exporter le modèle élagué, puis le quantifier selon `quant_type` dans `onnx_path` (`none`&nbsp;: une simple copie), en calibrant la quantification statique sur un lot d'images d'entraînement&nbsp;;
4. écrire dans `METRICS_PATH`, en JSON, les mesures du modèle flottant et du modèle réduit sur les images de test, sous les clés `float` et `compressed`.

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

from .data import IMAGE_SIZE, get_loader
from .model import LeNet
from .train import train_epoch

METRICS_PATH = Path("metrics/compress.json")


def export_onnx(model: nn.Module, path: str) -> None:
    example = (torch.rand(1, 3, *IMAGE_SIZE),)
    torch.onnx.export(
        model.eval(), example, path, dynamo=True, external_data=False, verbose=False
    )


class Calibration(CalibrationDataReader):
    def __init__(self, images: list[torch.Tensor]) -> None:
        self.images = iter(images)

    def get_next(self) -> dict | None:
        image = next(self.images, None)
        return None if image is None else {"x": image.numpy()}


def measure(path: str, images: np.ndarray, labels: np.ndarray) -> dict:
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    predictions, times = [], []
    for image in images:
        start = time.perf_counter()
        (logits,) = session.run(None, {"x": image[None]})
        times.append(time.perf_counter() - start)
        predictions.append(logits.argmax())
    return {
        "accuracy": float(np.mean(np.array(predictions) == labels)),
        "size_mb": Path(path).stat().st_size / 1e6,
        "gzip_mb": len(gzip.compress(Path(path).read_bytes())) / 1e6,
        "latency_ms": float(1000 * np.median(times[10:])),
    }


def load_images(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    images, labels = zip(*loader, strict=True)
    return torch.cat(images).numpy(), torch.cat(labels).numpy()


def compress(
    prepared_data: str,
    model_path: str,
    learning_rate: float,
    sparsity: float,
    quant_type: str,
    onnx_path: str,
) -> None:
    # Votre code ici
    ...
```

<details>
<summary>Conseils supplémentaires</summary>

Les exports et les quantifications écrivent des fichiers intermédiaires (le modèle flottant, le modèle élagué, celui que prépare `quant_pre_process`)&nbsp;: un dossier temporaire les garde, le temps de la fonction. L'élagage est celui des diapositives, avec `amount=sparsity`. `quantize_dynamic` et `quantize_static` lisent le modèle préparé par `quant_pre_process`, celui-ci avec `Calibration([image[None] for image in batch])` et `per_channel=True`, `batch` étant le premier lot de `train_loader`. `load_images(test_loader)` renvoie les images et les étiquettes que `measure` attend.

```python
def compress(
    prepared_data: str,
    model_path: str,
    learning_rate: float,
    sparsity: float,
    quant_type: str,
    onnx_path: str,
) -> None:
    torch.manual_seed(0)
    train_loader = get_loader(f"{prepared_data}/train", IMAGE_SIZE, shuffle=True)
    images, labels = load_images(get_loader(f"{prepared_data}/test", IMAGE_SIZE))
    model = LeNet()
    model.load_state_dict(...)
    with tempfile.TemporaryDirectory() as tmp_dir:
        float_path = f"{tmp_dir}/float.onnx"
        export_onnx(model, float_path)
        if sparsity > 0:
            ...  # élaguer, réentraîner une époque, rendre l'élagage définitif
        pruned_path = f"{tmp_dir}/pruned.onnx"
        export_onnx(model, pruned_path)
        if quant_type == "none":
            shutil.copyfile(...)
        else:
            preprocessed_path = f"{tmp_dir}/preprocessed.onnx"
            quant_pre_process(...)
            if quant_type == "dynamic":
                quantize_dynamic(...)
            else:
                batch, _ = next(iter(train_loader))
                quantize_static(...)
        metrics = {"float": measure(...), "compressed": measure(...)}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
```

</details>

<details>
<summary>Solution</summary>

`src/model_compression/compress.py`&nbsp;:

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

from .data import IMAGE_SIZE, get_loader
from .model import LeNet
from .train import train_epoch

METRICS_PATH = Path("metrics/compress.json")


def export_onnx(model: nn.Module, path: str) -> None:
    example = (torch.rand(1, 3, *IMAGE_SIZE),)
    torch.onnx.export(
        model.eval(), example, path, dynamo=True, external_data=False, verbose=False
    )


class Calibration(CalibrationDataReader):
    def __init__(self, images: list[torch.Tensor]) -> None:
        self.images = iter(images)

    def get_next(self) -> dict | None:
        image = next(self.images, None)
        return None if image is None else {"x": image.numpy()}


def measure(path: str, images: np.ndarray, labels: np.ndarray) -> dict:
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    predictions, times = [], []
    for image in images:
        start = time.perf_counter()
        (logits,) = session.run(None, {"x": image[None]})
        times.append(time.perf_counter() - start)
        predictions.append(logits.argmax())
    return {
        "accuracy": float(np.mean(np.array(predictions) == labels)),
        "size_mb": Path(path).stat().st_size / 1e6,
        "gzip_mb": len(gzip.compress(Path(path).read_bytes())) / 1e6,
        "latency_ms": float(1000 * np.median(times[10:])),
    }


def load_images(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    images, labels = zip(*loader, strict=True)
    return torch.cat(images).numpy(), torch.cat(labels).numpy()


def compress(
    prepared_data: str,
    model_path: str,
    learning_rate: float,
    sparsity: float,
    quant_type: str,
    onnx_path: str,
) -> None:
    torch.manual_seed(0)
    train_loader = get_loader(f"{prepared_data}/train", IMAGE_SIZE, shuffle=True)
    images, labels = load_images(get_loader(f"{prepared_data}/test", IMAGE_SIZE))
    model = LeNet()
    model.load_state_dict(torch.load(model_path))
    with tempfile.TemporaryDirectory() as tmp_dir:
        float_path = f"{tmp_dir}/float.onnx"
        export_onnx(model, float_path)
        if sparsity > 0:
            kinds = nn.Conv2d | nn.Linear
            weights = [(m, "weight") for m in model.modules() if isinstance(m, kinds)]
            prune.global_unstructured(
                weights, pruning_method=prune.L1Unstructured, amount=sparsity
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            train_epoch(model, train_loader, optimizer)
            for module, name in weights:
                prune.remove(module, name)
        pruned_path = f"{tmp_dir}/pruned.onnx"
        export_onnx(model, pruned_path)
        if quant_type == "none":
            shutil.copyfile(pruned_path, onnx_path)
        else:
            preprocessed_path = f"{tmp_dir}/preprocessed.onnx"
            quant_pre_process(pruned_path, preprocessed_path)
            if quant_type == "dynamic":
                quantize_dynamic(
                    preprocessed_path, onnx_path, weight_type=QuantType.QInt8
                )
            else:
                batch, _ = next(iter(train_loader))
                quantize_static(
                    preprocessed_path,
                    onnx_path,
                    Calibration([image[None] for image in batch]),
                    per_channel=True,
                )
        metrics = {
            "float": measure(float_path, images, labels),
            "compressed": measure(onnx_path, images, labels),
        }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
```

</details>

## L'étape de compression

Ajoutez à `params.yaml` `sparsity` (`0.5`), `quant_type` (`static`) et `onnx_path` (`landscape_classifier.onnx`), et à la chaîne l'étape `compress`, qui lance `python -m model_compression compress`, avec ses paramètres, ses dépendances, sa sortie et sa métrique, `metrics/compress.json`. Exécutez la chaîne, comparez avec `dvc metrics show` le modèle flottant et le modèle réduit, puis enregistrez le tout dans git. Le modèle réduit est-il plus petit&nbsp;? Plus rapide&nbsp;? Aussi exact&nbsp;?

<details>
<summary>Conseils supplémentaires</summary>

Les paramètres de `compress` sont les trois nouveaux, plus `prepared_data`, `model_path` et `learning_rate`, que la fonction lit aussi. Ses dépendances sont `prepared`, le modèle entraîné et le code qu'elle exécute&nbsp;: `__main__.py`, `compress.py`, et les modules qu'il importe.

À la fin de `params.yaml`&nbsp;:

```yaml
sparsity: ...
quant_type: ...
onnx_path: ...
```

```console
uv run dvc stage add -n compress -p ... -d ... -o ... -M ... python -m model_compression compress
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
sparsity: 0.5
quant_type: static
onnx_path: landscape_classifier.onnx
```

```console
uv run dvc stage add -n compress \
    -p prepared_data,model_path,learning_rate,sparsity,quant_type,onnx_path \
    -d prepared -d landscape_classifier.pt -d src/model_compression/__main__.py \
    -d src/model_compression/compress.py -d src/model_compression/data.py \
    -d src/model_compression/model.py -d src/model_compression/train.py \
    -o landscape_classifier.onnx -M metrics/compress.json \
    python -m model_compression compress
```

```console
uv run dvc repro
uv run dvc metrics show
git add .gitignore dvc.lock dvc.yaml metrics params.yaml src
git commit -m "Ajout de l'étape de compression"
```

Sur nos essais, le modèle élagué de moitié puis quantifié en int8 est trois fois plus petit (0,27 → 0,09&nbsp;Mo), à une exactitude d'un point plus basse (0,615 → 0,605)… mais plus lent&nbsp;: 0,7&nbsp;ms par image au lieu de 0,45. Les quantifications et déquantifications qu'ajoute la quantification statique (le format QDQ) coûtent plus que les calculs en int8 ne font gagner sur un CPU sans instructions int8 rapides (VNNI) et sur un modèle aussi petit. Un gain annoncé n'est pas un gain mesuré&nbsp;: mesurez sur le matériel cible.

</details>

### Expériences de compression

Lancez deux expériences&nbsp;: une quantification dynamique (`quant_type` à `dynamic`), et un élagage de 90&nbsp;% sans quantification (`sparsity` à `0.9`, `quant_type` à `none`). Comparez-les avec `dvc exp show`. Quel réglage donne le plus petit fichier&nbsp;? Le modèle le plus rapide&nbsp;? L'élagage seul réduit-il la taille du fichier&nbsp;? Et celle du fichier compressé&nbsp;?

<details>
<summary>Conseils supplémentaires</summary>

`-S` se répète pour changer plusieurs paramètres. Seule `compress` s'exécute.

```console
uv run dvc exp run -S ...
uv run dvc exp run -S ... -S ...
uv run dvc exp show --only-changed
```

</details>

<details>
<summary>Solution</summary>

```console
uv run dvc exp run -S quant_type=dynamic
uv run dvc exp run -S sparsity=0.9 -S quant_type=none
uv run dvc exp show --only-changed
```

Les deux quantifications donnent des fichiers trois fois plus petits, et la dynamique est la plus lente (2,5&nbsp;ms par image sur nos essais)&nbsp;: elle calcule les plages des activations à chaque inférence, et les convolutions en entiers d'ONNX Runtime (`ConvInteger`) sont lentes sur CPU. L'élagage seul ne réduit pas la taille du fichier&nbsp;: les zéros y sont stockés comme les autres poids, et les calculs ne vont pas plus vite. Mais le fichier compressé est cinq fois plus petit (0,23 → 0,05&nbsp;Mo)&nbsp;: les zéros se compressent bien, ce qui compte pour télécharger un modèle. Et l'exactitude ne baisse pas&nbsp;: l'époque de réentraînement compense l'élagage, signe que ce LeNet est surdimensionné pour ces images.

</details>

## Solution

Le projet terminé est sur la [branche `solution`](https://github.com/shuuchuu/compression/tree/solution) du dépôt.
