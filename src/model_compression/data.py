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
