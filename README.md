# Chaînes de traitements avec dvc et compression de modèles

## Initialisation du paquet Python avec `uv`

Utilisez `uv init --package .` pour créer un squelette de paquet Python.

Utilisez ensuite [`uv add`](https://docs.astral.sh/uv/concepts/projects/dependencies/#adding-dependencies) pour ajouter la dépendance `dvc` au projet.

<details><summary>Solution</summary>

    uv init --package .
    uv add dvc
</details>

## Initialisation du dépôt dvc

Initialisez le projet DVC avec [`dvc init`](https://dvc.org/doc/command-reference/init#init) puis enregistrez les changements dans git.

<details><summary>Solution</summary>

    dvc init
    git commit -m "Initialisation de DVC"
</details>

## Ajout des données à dvc

À l'aide de [`dvc import-url`](https://dvc.org/doc/command-reference/import-url), téléchargez le fichier zip suivant qui contient les données que nous allons utiliser&nbsp;:

    https://github.com/m09/dataset-landscape/archive/refs/heads/main.zip

Précisez le nom du fichier de sortie&nbsp;: `data.zip`.

Enregistrez ensuite les modifications avec `git`

<details><summary>Solution</summary>

    dvc import-url https://github.com/m09/dataset-landscape/archive/refs/heads/main.zip data.zip
    git add .gitignore data.zip.dvc

    git commit -m "Ajout des données"
</details>

## Création d'une étape de pipeline pour extraire les fichiers de l'archive zip

Créez une étape de pipeline DVC, à l'aide de `dvc stage add` ou en éditant vous-même le fichier `dvc.yaml`, afin de dézipper le fichier `data.zip` (vous pourrez utiliser pour cela la commande `unzip`)

<details><summary>Solution</summary>

    dvc stage add -n decompress -d data.zip -o dataset-landscape-main unzip data.zip
    dvc repro
    git add dvc.lock dvc.yaml .gitignore
    git commit -m "Ajout de l'étape de décompression"
</details>

## Préparation des données

Pour préparer les données avant d'entraîner puis de compresser des modèles, nous allons utiliser la fonction suivante, que vous pouvez intégrer dans les sources du projet python&nbsp;:

```python
import pathlib

import cv2
import numpy
import sklearn.metrics
import sklearn.utils

CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
CLASS_INDICES = {l: i for i, l in enumerate(CLASS_NAMES)}


def get_images(
    dir_path: pathlib.Path, image_size: tuple[int, int], shuffle: bool = True
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    images = []
    labels = []
    file_paths = []

    # On itère sur les sous-dossier de la racine : ils correspondent chacun à une
    # classe
    for subdir_path in dir_path.iterdir():

        # Attribuez le bon label en fonction du nom du dossier "labels"
        # Votre code ici
        label = CLASS_INDICES.get(subdir_path.name)

        # On ajoute chaque image du label (dossier) courant à notre dataset
        for image_path in subdir_path.iterdir():
            # Utilisation de OpenCV pour charger l'image
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, image_size)
            image = image.astype("float32")
            images.append(image)
            labels.append(label)
            file_paths.append(image_path)
    images_array = numpy.array(images)
    labels_array = numpy.array(labels)
    file_paths_array = numpy.array(file_paths)

    # Mélange de ces tableaux
    if shuffle:
        images_array, labels_array, file_paths_array = sklearn.utils.shuffle(
            images_array, labels_array, file_paths_array
        )
    return images_array, labels_array, file_paths_array
```

## Entraînement d'un modèle simple

Nous allons maintenant entraîner un modèle de classification d'images&nbsp;: le modèle historique LeNet. En voici une implémentation. Adaptez-la pour enregistrer le modèle appris avec `dvclive` et intégrez-la au dossier de sources Python.

### Modèle

```python
import tensorflow as tf


def get_lenet(
    image_size: tuple[int, int], learning_rate: float = 1e-4
) -> tf.keras.models.Model:
    def conv(filters: int, padding: str) -> tf.keras.layers.Conv2D:
        return tf.keras.layers.Conv2D(
            filters=filters, kernel_size=5, padding=padding, activation="sigmoid"
        )

    def pooling() -> tf.keras.layers.MaxPooling2D:
        return tf.keras.layers.MaxPooling2D()

    def dense(units: int, activation: str = "sigmoid") -> tf.keras.layers.Dense:
        return tf.keras.layers.Dense(units, activation=activation)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(*image_size, 3)),
            conv(6, "same"),
            pooling(),
            conv(16, "valid"),
            pooling(),
            tf.keras.layers.Flatten(),
            dense(120),
            dense(84),
            dense(6, activation="softmax"),
        ],
        name="le_net",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
```

### Entraînement

```python
from pathlib import Path

from ... import get_images  # À modifier en fonction des noms des fichiers créés
from ... import get_lenet  # À modifier en fonction des noms des fichiers créés


def train(
    data_dir: str,
    image_size: tuple[int, int],
    learning_rate: float,
) -> None:
    images, labels, paths = get_images(Path(data_dir), image_size)
    model = get_lenet(image_size, learning_rate)
    model.fit(images, labels, 128, epochs=3)
    model.save("landscape_classifier.keras")
```

### Définition d'une étape de traitement

Définissez maintenant une [étape de traitement](https://dvc.org/doc/command-reference/stage/add) qui lance l'entraînement du modèle.

<details><summary>Solution</summary>

Exemple, qui peut varier en fonction de l'organisation précise du code&nbsp;:

    dvc stage add -n train \
                  -d dataset-landscape-main/seg_train \
                  -d tp_compression \
                  -o landscape_classifier.keras \
                  tp-compression

    dvc repro
</details>

## Compression de modèle a posteriori

Démonstration d'ajout d'un code de compression qui utilise les recommandations du [guide de quantification TensorFlow](https://www.tensorflow.org/model_optimization/guide/quantization/post_training).

## Solution

Dépôt [GitHub](https://dagshub.com/m09/compression) sur la branche `solution`.
