from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy
import tensorflow as tf

from .core.data import get_images


def _evaluate_tflite_model(
    tflite_model: Any, test_images: numpy.ndarray, test_labels: numpy.ndarray
) -> float:
    # Initialize TFLite interpreter using the model.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_tensor_index = interpreter.get_input_details()[0]["index"]
    output = interpreter.tensor(interpreter.get_output_details()[0]["index"])

    # Run predictions on every image in the "test" dataset.
    prediction_digits = []
    for test_image in test_images:
        # Pre-processing: add batch dimension and convert to float32 to match with
        # the model's input data format.
        test_image = numpy.expand_dims(test_image, axis=0)
        interpreter.set_tensor(input_tensor_index, test_image)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        digit = numpy.argmax(output()[0])
        prediction_digits.append(digit)

    # Compare prediction results with ground truth labels to calculate accuracy.
    accurate_count = 0
    for index in range(len(prediction_digits)):
        if prediction_digits[index] == test_labels[index]:
            accurate_count += 1
    return accurate_count * 1.0 / len(prediction_digits)


def compress(
    train_data: str,
    test_data: str,
    model_path: str,
    tflite_model_path: str,
    image_size: tuple[int, int],
) -> None:
    model = tf.keras.models.load_model(model_path)

    def representative_dataset_gen() -> Iterator[list[numpy.ndarray]]:
        images, _labels = get_images(Path(train_data), image_size, split=False)
        images = images[
            numpy.random.choice(images.shape[0], size=1_000, replace=False)
        ].astype("float32")
        for i in range(images.shape[0]):
            # Get sample input data as a numpy array in a method of your choosing.
            yield [images[[i]]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    tflite_quant_model = converter.convert()
    Path(tflite_model_path).write_bytes(tflite_quant_model)
    test_images, test_labels = get_images(Path(test_data), image_size, split=False)
    print(
        _evaluate_tflite_model(
            tflite_quant_model, test_images.astype("float32"), test_labels
        )
    )
