import pathlib
from typing import BinaryIO, Literal, overload

import numpy
import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

LABEL_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABEL_NAMES)}


def process_image(
    file: BinaryIO | str | pathlib.Path, image_size: tuple[int, int]
) -> numpy.ndarray:
    return numpy.array(Image.open(file).resize(image_size))[None, ...]


@overload
def get_images(
    dir_path: pathlib.Path, image_size: tuple[int, int], split: Literal[True] = True
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]: ...
@overload
def get_images(
    dir_path: pathlib.Path, image_size: tuple[int, int], split: Literal[False]
) -> tuple[numpy.ndarray, numpy.ndarray]: ...
def get_images(
    dir_path: pathlib.Path, image_size: tuple[int, int], split: bool = True
) -> (
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
    | tuple[numpy.ndarray, numpy.ndarray]
):
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

    return (
        train_test_split(images_array, labels_array, test_size=0.3, shuffle=True)
        if split
        else (images_array, labels_array)
    )
