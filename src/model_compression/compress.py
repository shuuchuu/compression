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

    def get_next(self) -> dict | None:  # ty: ignore[invalid-method-override]
        image = next(self.images, None)
        return None if image is None else {"x": image.numpy()}


def measure(path: str, images: np.ndarray, labels: np.ndarray) -> dict:
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    predictions, times = [], []
    for image in images:
        start = time.perf_counter()
        (logits,) = session.run(None, {"x": image[None]})
        times.append(time.perf_counter() - start)
        predictions.append(logits.argmax())  # ty: ignore[unresolved-attribute]
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
