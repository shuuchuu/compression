from pathlib import Path

from .data import get_images
from .model import get_lenet


def train(
    data_dir: str, image_size: tuple[int, int], learning_rate: float, model_path: str
) -> None:
    X_train, X_val, y_train, y_val = get_images(Path(data_dir), image_size, split=True)
    model = get_lenet(image_size, learning_rate)
    model.fit(X_train, y_train, 128, validation_data=(X_val, y_val), epochs=3)
    model.save(model_path)
