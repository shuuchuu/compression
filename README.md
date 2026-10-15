# Pipelines DVC et compression de modèles

*[English version](https://github.com/shuuchuu/compression/tree/main-en)*

Le projet des travaux pratiques *Pipelines DVC et compression de modèles*. Leur énoncé est
ci-dessous, pour les faire dans un Codespace GitHub ou un clone local&nbsp;; c'est le même
que celui du notebook des travaux pratiques, en
[français](https://colab.research.google.com/github/shuuchuu/labs/blob/main/compression-hands-on-fr.ipynb)
ou en
[anglais](https://colab.research.google.com/github/shuuchuu/labs/blob/main/compression-hands-on-en.ipynb),
qui les fait dans Colab.

Nous allons construire, avec DVC, la chaîne de traitements d'un classifieur d'images de
paysages&nbsp;: préparation des données, entraînement, puis compression du modèle.

## Mise en place de l'environnement de travail

Ouvrez le dépôt dans un Codespace&nbsp;: sur GitHub, **Code** > **Codespaces** > **Create
codespace on main**, ou [ce lien](https://codespaces.new/shuuchuu/compression). Dans un
clone local, installez [uv](https://docs.astral.sh/uv/getting-started/installation/).

Les commandes ci-dessous se lancent dans un terminal à la racine du dépôt (**Terminal** >
**Nouveau terminal** dans le Codespace). DVC s'y lance avec `uv run`, qui l'exécute dans
l'environnement du projet.

## Création du paquet Python

Créez un paquet Python `compression` dans le dépôt avec `uv init`, puis ajoutez-lui avec
`uv add` DVC et les bibliothèques qu'utilise le code fourni plus loin&nbsp;: `numpy`,
`pillow`, `scikit-learn`, `tensorflow` et `tqdm`. Enregistrez ensuite le projet dans git.

**Documentation utile**

- [`uv init`](https://docs.astral.sh/uv/reference/cli/#uv-init)
- [`uv add`](https://docs.astral.sh/uv/reference/cli/#uv-add)

<details>
<summary>Conseils supplémentaires</summary>

`uv init --package --name compression .` crée le paquet dans le dossier courant, ses
sources dans `src/compression/`. `uv add` prend plusieurs paquets à la fois, et crée
l'environnement du projet, `.venv`, que git ignore.

```console
uv init --package --name compression .
uv add dvc ...
git add .
git commit -m "..."
```

</details>

<details>
<summary>Solution</summary>

```console
uv init --package --name compression .
uv add dvc numpy pillow scikit-learn tensorflow tqdm
git add .
git commit -m "Création du paquet Python"
```

</details>

## Initialisation du dépôt DVC

Initialisez DVC dans le dépôt, puis enregistrez dans git les fichiers qu'il crée.

**Documentation utile**

- [`dvc init`](https://doc.dvc.org/command-reference/init)

<details>
<summary>Conseils supplémentaires</summary>

`dvc init` crée le dossier `.dvc/` et ajoute lui-même ses fichiers à l'index de git&nbsp;:
il ne reste qu'à les commiter.

```console
uv run dvc init
git commit -m "..."
```

</details>

<details>
<summary>Solution</summary>

```console
uv run dvc init
git commit -m "Initialisation de DVC"
```

</details>

## Ajout des données à DVC

Avec `dvc import`, téléchargez le dossier `landscape` du dépôt
`https://github.com/shuuchuu/datasets` dans le dossier `data`, puis enregistrez dans git
les fichiers qui décrivent l'import. DVC clone le dépôt pour cela&nbsp;: comptez quelques
minutes.

`data/seg_train/` et `data/seg_test/` contiennent les images d'entraînement et de test,
un dossier par classe (`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`)&nbsp;;
`data/seg_pred/`, des images sans étiquette, ne sert pas ici.

**Documentation utile**

- [`dvc import`](https://doc.dvc.org/command-reference/import)

<details>
<summary>Conseils supplémentaires</summary>

`dvc import` prend l'adresse du dépôt, le chemin du dossier dans ce dépôt, et le dossier
de sortie avec `-o`. Il écrit `data.dvc`, qui retient le commit importé (`dvc update`
irait chercher le dernier), et ajoute `/data` à `.gitignore`&nbsp;: ce sont ces deux
fichiers qui vont dans git, pas les images.

```console
uv run dvc import https://github.com/shuuchuu/datasets ... -o ...
git add ...
git commit -m "..."
```

</details>

<details>
<summary>Solution</summary>

```console
uv run dvc import https://github.com/shuuchuu/datasets landscape -o data
git add .gitignore data.dvc
git commit -m "Ajout des données"
```

</details>

## Préparation des données

L'entraînement et la compression lisent les mêmes 17&nbsp;000 images. Plutôt que de les
décoder et de les redimensionner à chaque fois, une première étape de la chaîne,
`prepare`, le fait une fois pour toutes et enregistre le résultat sous forme de tableaux,
dans `prepared/train.npz` et `prepared/test.npz`. Voici son code, à mettre dans
`src/compression/data.py`&nbsp;:

```python
from pathlib import Path

import numpy
import tqdm
from PIL import Image

LABEL_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]


def load_images(
    dir_path: Path, image_size: tuple[int, int]
) -> tuple[numpy.ndarray, numpy.ndarray]:
    images = []
    labels = []
    for label, name in enumerate(LABEL_NAMES):
        for image_path in tqdm.tqdm(sorted((dir_path / name).iterdir()), desc=name):
            image = Image.open(image_path).convert("RGB").resize(tuple(image_size))
            images.append(numpy.asarray(image))
            labels.append(label)
    return numpy.stack(images), numpy.array(labels)


def prepare(
    train_data: str, test_data: str, image_size: tuple[int, int], prepared_data: str
) -> None:
    output_dir = Path(prepared_data)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data_dir in (("train", train_data), ("test", test_data)):
        images, labels = load_images(Path(data_dir), image_size)
        numpy.savez(output_dir / f"{name}.npz", images=images, labels=labels)


def load_prepared(
    prepared_data: str, name: str
) -> tuple[numpy.ndarray, numpy.ndarray]:
    with numpy.load(Path(prepared_data) / f"{name}.npz") as arrays:
        return arrays["images"], arrays["labels"]
```

Et le point d'entrée de la chaîne, `src/compression/__main__.py`&nbsp;:
`python -m compression prepare` lance l'étape `prepare`, avec les paramètres que
`dvc.yaml` lui déclare, lus dans `params.yaml` par `dvc.api.params_show`.

```python
from sys import argv

from dvc.api import params_show

from .data import prepare

STAGES = {"prepare": prepare}

if len(argv) != 2 or argv[1] not in STAGES:
    msg = f"Usage: python -m compression {{{','.join(STAGES)}}}"
    raise SystemExit(msg)

STAGES[argv[1]](**params_show(stages=[argv[1]]))
```

Ajoutez à la chaîne l'étape `prepare`&nbsp;:

1. Créez `params.yaml`, à la racine du dépôt, avec les paramètres de `prepare`&nbsp;:
   `train_data` (`data/seg_train`), `test_data` (`data/seg_test`), `image_size`
   (`[100, 100]`) et `prepared_data` (`prepared`).
2. Avec `dvc stage add`, créez l'étape `prepare`, qui lance `python -m compression
   prepare`, en déclarant ses paramètres, ses dépendances (les images et le code qui les
   lit) et sa sortie.
3. Exécutez la chaîne avec `dvc repro`, puis enregistrez dans git le code, `params.yaml`
   et ce que DVC a écrit.

**Documentation utile**

- [`dvc stage add`](https://doc.dvc.org/command-reference/stage/add)
- [`dvc repro`](https://doc.dvc.org/command-reference/repro)
- [`dvc.api.params_show`](https://doc.dvc.org/api-reference/params_show)

<details>
<summary>Conseils supplémentaires</summary>

Les options de `dvc stage add`&nbsp;: `-n` nomme l'étape, `-p` déclare ses paramètres
(leurs noms dans `params.yaml`, séparés par des virgules), `-d` chaque dépendance, `-o`
sa sortie, puis vient la commande. L'étape est écrite dans `dvc.yaml`. `dvc repro` lance
les étapes dont une dépendance ou un paramètre a changé, et retient dans `dvc.lock` les
empreintes de ce qu'elles ont lu et produit&nbsp;; DVC ajoute aussi `/prepared` à
`.gitignore`.

```yaml
train_data: data/seg_train
test_data: ...
image_size: ...
prepared_data: ...
```

```console
uv run dvc stage add -n prepare -p ... -d ... -o ... python -m compression prepare
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
image_size: [100, 100]
prepared_data: prepared
```

```console
uv run dvc stage add -n prepare \
    -p train_data,test_data,image_size,prepared_data \
    -d data/seg_train -d data/seg_test -d src/compression/data.py \
    -o prepared \
    python -m compression prepare
uv run dvc repro
git add .gitignore dvc.lock dvc.yaml params.yaml src
git commit -m "Ajout de l'étape de préparation"
```

</details>

## Entraînement d'un modèle simple

Nous allons maintenant entraîner un modèle de classification d'images&nbsp;: le modèle
historique LeNet. Voici son code, à mettre dans `src/compression/model.py`&nbsp;:

```python
import keras


def get_lenet(
    image_size: tuple[int, int], learning_rate: float = 1e-4
) -> keras.models.Model:
    def conv(filters: int, padding: str) -> keras.layers.Conv2D:
        return keras.layers.Conv2D(
            filters=filters, kernel_size=5, padding=padding, activation="sigmoid"
        )

    def pooling() -> keras.layers.MaxPooling2D:
        return keras.layers.MaxPooling2D()

    def dense(units: int, activation: str = "sigmoid") -> keras.layers.Dense:
        return keras.layers.Dense(units, activation=activation)

    model = keras.Sequential(
        [
            keras.layers.Input((*image_size, 3)),
            conv(6, "same"),
            pooling(),
            conv(16, "valid"),
            pooling(),
            keras.layers.Flatten(),
            dense(120),
            dense(84),
            dense(6, activation="softmax"),
        ],
        name="le_net",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
```

Celui de l'entraînement, dans `src/compression/train.py`, qui garde 30&nbsp;% des images
d'entraînement pour la validation&nbsp;:

```python
from sklearn.model_selection import train_test_split

from .data import load_prepared
from .model import get_lenet


def train(prepared_data: str, learning_rate: float, model_path: str) -> None:
    images, labels = load_prepared(prepared_data, "train")
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.3, random_state=0
    )
    model = get_lenet(images.shape[1:3], learning_rate)
    model.fit(X_train, y_train, 128, validation_data=(X_val, y_val), epochs=2)
    model.save(model_path)
```

Et dans `src/compression/__main__.py`, l'étape `train`&nbsp;:

```python
from .data import prepare
from .train import train

STAGES = {"prepare": prepare, "train": train}
```

Ajoutez à la chaîne l'étape `train`, qui lance `python -m compression train`&nbsp;: ses
paramètres `learning_rate` (`0.0001`) et `model_path` (`landscape_classifier.keras`)
s'ajoutent à `params.yaml`, et elle lit aussi `prepared_data`. Exécutez la chaîne, puis
enregistrez le tout dans git.

<details>
<summary>Conseils supplémentaires</summary>

Les dépendances de `train` sont la sortie de `prepare` et le code qu'elle exécute&nbsp;:
c'est en déclarant `prepared` comme dépendance que DVC sait que `train` vient après
`prepare`. Sa sortie est le modèle. `dvc repro` ne relance pas `prepare`, qui est à jour.

```yaml
learning_rate: ...
model_path: ...
```

```console
uv run dvc stage add -n train -p ... -d prepared -d ... -o ... python -m compression train
uv run dvc repro
git add ...
git commit -m "..."
```

</details>

<details>
<summary>Solution</summary>

À la fin de `params.yaml`&nbsp;:

```yaml
learning_rate: 0.0001
model_path: landscape_classifier.keras
```

```console
uv run dvc stage add -n train \
    -p prepared_data,learning_rate,model_path \
    -d prepared -d src/compression/data.py -d src/compression/model.py \
    -d src/compression/train.py \
    -o landscape_classifier.keras \
    python -m compression train
uv run dvc repro
git add .gitignore dvc.lock dvc.yaml params.yaml src
git commit -m "Ajout de l'étape d'entraînement"
```

</details>

### Changement d'un paramètre

Passez `learning_rate` à `0.001` dans `params.yaml`. Avant de relancer la chaîne, voyez
avec `dvc status` les étapes que ce changement rend obsolètes, et avec `dvc params diff`
ce qui a changé depuis le dernier commit. Relancez-la ensuite&nbsp;: quelles étapes
s'exécutent&nbsp;? Et lesquelles s'exécuteraient si vous changiez `image_size`&nbsp;?

**Documentation utile**

- [`dvc status`](https://doc.dvc.org/command-reference/status)
- [`dvc params diff`](https://doc.dvc.org/command-reference/params/diff)

<details>
<summary>Conseils supplémentaires</summary>

`dvc status` compare les dépendances et les paramètres de chaque étape avec les
empreintes de `dvc.lock`. Pour `image_size`, regardez de quoi dépend `train`.

```console
uv run dvc status
uv run dvc params diff
uv run dvc repro
```

</details>

<details>
<summary>Solution</summary>

```console
sed -i "s/^learning_rate: .*/learning_rate: 0.001/" params.yaml
uv run dvc status
uv run dvc params diff
uv run dvc repro
```

Seule `train` est obsolète et s'exécute&nbsp;: `prepare` ne lit pas `learning_rate`.
Changer `image_size` rendrait `prepare` obsolète, et `train` s'exécuterait après elle,
parce que sa dépendance `prepared` aurait changé. Si `prepare` produisait exactement les
mêmes fichiers, DVC ne relancerait pas `train`&nbsp;: il compare les empreintes, pas les
dates.

</details>

## Compression du modèle a posteriori

Démonstration de l'ajout d'une étape `compress`, qui exporte le modèle au format ONNX et
le quantifie en entiers sur 8&nbsp;bits, en calibrant la quantification sur des images
d'entraînement, puis mesure sa précision sur les images de test&nbsp;: elle lit le modèle
et `prepared/`.

## Solution

Le code de la solution est sur la
[branche `solution`](https://github.com/shuuchuu/compression/tree/solution) du dépôt.
