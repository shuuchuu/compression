from pathlib import Path

from .core.data import get_images
from .core.model import get_lenet


def train(
    train_data: str, image_size: tuple[int, int], learning_rate: float, model_path: str
) -> None:
    X_train, X_val, y_train, y_val = get_images(Path(train_data), image_size)
    model = get_lenet(image_size, learning_rate)
    model.fit(X_train, y_train, 128, validation_data=(X_val, y_val), epochs=2)
    model.save(model_path)
