# Chaînes de traitements avec dvc et compression de modèles

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/shuuchuu/compression)

Dans le codespace, presser Ctrl-k puis v avec le fichier README.md ouvert pour le visualiser correctement.

## Initialisation du paquet Python avec `uv`

Utilisez `uv init --name compression --package .` pour créer un squelette de paquet Python.

Utilisez ensuite [`uv add`](https://docs.astral.sh/uv/concepts/projects/dependencies/#adding-dependencies) pour ajouter la dépendance `dvc` au projet.

<details><summary>Solution</summary>

    uv init --name compression --package .
    uv add dvc

</details>

## Initialisation du dépôt dvc

Initialisez le projet DVC avec [`dvc init`](https://dvc.org/doc/command-reference/init#init) puis enregistrez les changements dans git.

<details><summary>Solution</summary>

    dvc init
    git commit -m "Initialisation de DVC"

</details>

## Ajout des données à dvc

À l'aide de [`dvc import`](https://doc.dvc.org/command-reference/import), téléchargez le dossier `landscape` du dépôt suivant&nbsp;:

    https://github.com/shuuchuu/datasets

Précisez le nom du dossier de sortie&nbsp;: `data`.

Enregistrez ensuite les modifications avec `git`

<details><summary>Solution</summary>

    dvc import https://github.com/shuuchuu/datasets landscape -o data
    git add .gitignore data.dvc

    git commit -m "Ajout des données"

</details>

## Préparation des données

Pour préparer les données avant d'entraîner puis de compresser des modèles, nous allons utiliser la fonction suivante, que vous pouvez intégrer dans les sources du projet python&nbsp;:

```python
import pathlib
import typing

import numpy
import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

LABEL_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABEL_NAMES)}


def process_image(
    file: typing.BinaryIO | str | pathlib.Path, image_size: tuple[int, int]
) -> numpy.ndarray:
    return numpy.array(Image.open(file).resize(image_size))[None, ...]


def get_images(
    dir_path: pathlib.Path, image_size: tuple[int, int]
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    images = []
    labels = []

    for subdir_path in tqdm.tqdm(
        list(dir_path.iterdir()), desc="Traitement des dossiers"
    ):
        dir_name = subdir_path.name

        label = LABEL_TO_INDEX.get(dir_name)

        for image_path in tqdm.tqdm(
            list(subdir_path.iterdir()), desc=f"Dossier {dir_name}", leave=False
        ):
            images.append(process_image(image_path, image_size))
            labels.append(label)

    images_array = numpy.vstack(images)
    labels_array = numpy.array(labels)

    return train_test_split(images_array, labels_array, test_size=0.3, shuffle=True)
```

## Entraînement d'un modèle simple

Nous allons maintenant entraîner un modèle de classification d'images&nbsp;: le modèle historique LeNet. En voici une implémentation. Adaptez-la pour enregistrer le modèle appris avec `dvc` et intégrez-la au dossier de sources Python.

### Modèle

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

### Entraînement

```python
from pathlib import Path

from .data import get_images  # À adapter en fonction du nom de vos fichiers
from .model import get_lenet  # À adapter en fonction du nom de vos fichiers


def train(
    data_dir: str, image_size: tuple[int, int], learning_rate: float, model_path: str
) -> None:
    X_train, X_val, y_train, y_val = get_images(Path(data_dir), image_size)
    model = get_lenet(image_size, learning_rate)
    model.fit(X_train, y_train, 128, validation_data=(X_val, y_val), epochs=3)
    model.save(model_path)
```

### Définition d'une étape de traitement

Définissez maintenant une [étape de traitement](https://dvc.org/doc/command-reference/stage/add) qui lance l'entraînement du modèle.

<details><summary>Solution</summary>

Exemple, qui peut varier en fonction de l'organisation précise du code&nbsp;:

    dvc stage add -n train \
                  -d data/seg_train \
                  -d src/compression \
                  -o landscape_classifier.keras \
                  compression

    dvc repro

</details>

## Compression de modèle a posteriori

Démonstration d'ajout d'un code de compression qui utilise un export ONNX.

## Solution

Dépôt [GitHub](https://github.com/shuuchuu/compression) sur la branche `solution`.
