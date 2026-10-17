# DVC Pipelines and Model Compression

*[Version française](https://github.com/shuuchuu/compression)*

The project of the *DVC pipelines and model compression* lab. Its instructions are below, to run the lab in a GitHub Codespace or a local clone; they are the same as the lab's notebook, in [English](https://colab.research.google.com/github/shuuchuu/labs/blob/main/compression-hands-on-en.ipynb) or in [French](https://colab.research.google.com/github/shuuchuu/labs/blob/main/compression-hands-on-fr.ipynb), which runs it in Colab.

We are going to build, with DVC, the pipeline of a landscape image classifier, in PyTorch: data preparation, training, then model compression, each stage measured and compared from one experiment to the next.

## Setting up the Working Environment

This English version of the lab is the repository's `main-en` branch. Open it in a Codespace: on GitHub, on the `main-en` branch, **Code** > **Codespaces** > **Create codespace on main-en**, or [this link](https://codespaces.new/shuuchuu/compression/tree/main-en). In a local clone (`git clone -b main-en https://github.com/shuuchuu/compression.git`), install [uv](https://docs.astral.sh/uv/getting-started/installation/).

The commands below run in a terminal at the root of the repository (**Terminal** > **New Terminal** in the Codespace). The files whose code is given are to be created where stated.

## Creating the Python package and the DVC repository

Create a Python package `model_compression` with `uv init`. The PyTorch wheels published on PyPI embed the CUDA libraries, several gigabytes, useless here, where everything runs on a CPU: before installing anything, append to `pyproject.toml` the index of PyTorch's CPU wheels, which only serves (`explicit = true`) the packages `[tool.uv.sources]` assigns to it:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
torchvision = { index = "pytorch-cpu" }
```

Then add to the package, with `uv add`, DVC and the libraries the code given further down uses: `torch`, `torchvision`, `onnx`, `onnxscript` (which PyTorch's ONNX export needs) and `onnxruntime`. Finally, initialize DVC in the repository with `dvc init`, then save everything in git. DVC runs with `uv run`, which runs it in the project's environment.

**Useful documentation**

- [`uv init`](https://docs.astral.sh/uv/reference/cli/#uv-init)
- [`uv add`](https://docs.astral.sh/uv/reference/cli/#uv-add)
- [`dvc init`](https://doc.dvc.org/command-reference/init)

<details>
<summary>Additional hints</summary>

`uv init --package --name model_compression .` creates the package in the current directory, its sources in `src/model_compression/` (not `compression`: Python 3.14's standard library has a package by that name, which would hide ours). `uv add` takes several packages at once, and creates the project's environment, `.venv`, which git ignores. `dvc init` creates the `.dvc/` directory and adds its files to git's index itself.

```console
uv init --package --name model_compression .
```

At the end of `pyproject.toml`:

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

At the end of `pyproject.toml`:

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
git commit -m "Create the Python package and the DVC repository"
```

</details>

## Adding the data to DVC

The following commands download into `data/` the `landscape` directory of the [shuuchuu/datasets](https://github.com/shuuchuu/datasets/tree/main/landscape) repository, without `data/seg_pred/`, unlabeled images that aren't used here. `data/seg_train/` and `data/seg_test/` hold the training and test images (150×150 pixels), one directory per class (`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`).

```console
curl -sSL https://github.com/shuuchuu/datasets/raw/refs/heads/main/ghsparse.sh | bash -s landscape
mv landscape data
rm -r data/seg_pred
```

Add `data/` to DVC, then save in git the files that describe it.

<details>
<summary>Additional hints</summary>

`dvc add` puts the directory in DVC's cache, `.dvc/cache`, and writes `data.dvc`, which holds its md5 hash; it also adds `/data` to `.gitignore`: these two files go in git, not the images.

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
git commit -m "Add the data"
```

</details>

## Preparing the data

Training and compression read the same images. Rather than decoding and resizing them every time, a first stage of the pipeline, `prepare`, does it once and for all: it copies the first `images_per_class` images of each class into `prepared/train/` and `prepared/test/`, resized to `IMAGE_SIZE`, 88×88 pixels. 500 images per class are enough for this lab, and a training on a CPU then takes one to two minutes. At this size, the last feature maps of the training's LeNet are 20×20, which its `AdaptiveAvgPool2d` layer reduces to 5×5 by averaging regular windows: the ONNX export turns that into a single operation (at 100×100, into about a hundred).

Its code, in `src/model_compression/data.py`, also turns the images into tensors, their pixels scaled to [0, 1] (`get_transform`), and loads them in batches of 64 (`get_loader`):

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

The pipeline's entry point, `src/model_compression/__main__.py`: `python -m model_compression <stage>` runs that stage's function, `prepare` from `data.py`, then `train` from `train.py` and `compress` from `compress.py`, which you'll write further down, with the parameters `dvc.yaml` declares for it, read from `params.yaml` by `dvc.api.params_show`. A stage's module is only imported when it runs: this file won't change anymore.

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

Add the `prepare` stage to the pipeline:

1. Create `params.yaml`, at the root of the repository, with `prepare`'s parameters: `train_data` (`data/seg_train`), `test_data` (`data/seg_test`), `images_per_class` (`500`) and `prepared_data` (`prepared`).
2. With `dvc stage add`, create the `prepare` stage, which runs `python -m model_compression prepare`, declaring its parameters, its dependencies (the images and the code it runs) and its output.
3. Run the pipeline with `dvc repro`, then save in git the code, `params.yaml` and what DVC wrote.

**Useful documentation**

- [`dvc stage add`](https://doc.dvc.org/command-reference/stage/add)
- [`dvc repro`](https://doc.dvc.org/command-reference/repro)
- [`dvc.api.params_show`](https://doc.dvc.org/api-reference/params_show)

<details>
<summary>Additional hints</summary>

The options of `dvc stage add`: `-n` names the stage, `-p` declares its parameters (their names in `params.yaml`, separated by commas), `-d` each dependency, `-o` its output, then comes the command. The code `prepare` runs is `__main__.py` and `data.py`. The stage is written to `dvc.yaml`. `dvc repro` runs the stages whose dependencies or parameters changed, and records in `dvc.lock` the hashes of what they read and produced; DVC also adds `/prepared` to `.gitignore`.

`params.yaml`:

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

`params.yaml`:

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
git commit -m "Add the preparation stage"
```

</details>

## Remote storage

DVC's cache, `.dvc/cache`, only exists on this machine. To share data and models, or to get them back elsewhere, DVC sends them to a remote storage, declared once and for all in `.dvc/config`. Here, a plain directory, `/tmp/dvc-store`, plays that part.

1. With `dvc remote add`, declare the `local` storage, in `/tmp/dvc-store`, as the default storage, then save the configuration in git.
2. Send the data there with `dvc push`, and look at what `/tmp/dvc-store` holds.
3. Delete `prepared/`, then get it back with `dvc pull`.

What would it take, on another machine, to get `prepared/` back without running `prepare` again?

**Useful documentation**

- [`dvc remote add`](https://doc.dvc.org/command-reference/remote/add)
- [`dvc push`](https://doc.dvc.org/command-reference/push)
- [`dvc pull`](https://doc.dvc.org/command-reference/pull)

<details>
<summary>Additional hints</summary>

`-d` makes the storage the default one, which `dvc push` and `dvc pull` use without an option. `dvc remote add` writes to `.dvc/config`, which git tracks. The storage files each file under its md5 hash, as the cache does (`ls /tmp/dvc-store/files/md5`).

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
git commit -m "Add the remote storage"
uv run dvc push
ls /tmp/dvc-store/files/md5 | head
rm -r prepared
uv run dvc pull
ls prepared
```

`dvc push` sends the content of the cache that `data.dvc` and `dvc.lock` describe: the images of `data/` and those of `prepared/`, each under its hash. Here, `dvc pull` downloads nothing, since the cache still has the files: it puts them back in place (`dvc checkout`). On another machine, `git clone` then `dvc pull` would download `prepared/` from the storage, without computing anything again, if it were shared (an S3 bucket, an SSH server…): a directory of `/tmp` isn't.

</details>

## Training a simple model

We are now going to train an image classification model: the historical LeNet model, two convolutions then three dense layers. Here is its code, to put in `src/model_compression/model.py`:

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

And the training's, in `src/model_compression/train.py`: `train_epoch` goes once over the training images, `evaluate` measures the accuracy (the share of images classified right), and `train` runs `epochs` epochs, then saves the model's weights to `model_path` and its accuracy on the test images to `metrics/train.json`, a metrics file DVC can read and compare:

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

Add the `train` stage to the pipeline, which runs `python -m model_compression train`: its parameters `learning_rate` (`0.001`), `epochs` (`5`) and `model_path` (`landscape_classifier.pt`) are added to `params.yaml`, and it also reads `prepared_data`. Declare `metrics/train.json` as the stage's metric, run the pipeline, show the metrics with `dvc metrics show`, then save everything in git.

**Useful documentation**

- [`dvc metrics show`](https://doc.dvc.org/command-reference/metrics/show)

<details>
<summary>Additional hints</summary>

The dependencies of `train` are `prepare`'s output and the code it runs (`__main__.py`, `data.py`, `model.py` and `train.py`): declaring `prepared` as a dependency is how DVC knows that `train` comes after `prepare`. Its output is the model (`-o`). Its metric is declared with `-M`: a file DVC doesn't cache and git tracks, to compare the metrics from one commit to the next. `dvc repro` doesn't run `prepare` again, which is up to date. The training takes one to two minutes.

At the end of `params.yaml`:

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

At the end of `params.yaml`:

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
git commit -m "Add the training stage"
```

</details>

### An experiment: changing the learning rate

Run with `dvc exp run` an experiment where `learning_rate` is `0.003`, without editing `params.yaml` by hand. Compare its metrics with the last commit's with `dvc metrics diff`, then list the experiments with `dvc exp show`. Which stages did the experiment run? Which ones would run if you changed `images_per_class`? If the accuracy improves, keep this rate: save in git what the experiment changed.

**Useful documentation**

- [`dvc exp run`](https://doc.dvc.org/command-reference/exp/run)
- [`dvc metrics diff`](https://doc.dvc.org/command-reference/metrics/diff)
- [`dvc exp show`](https://doc.dvc.org/command-reference/exp/show)

<details>
<summary>Additional hints</summary>

`-S learning_rate=0.003` changes the parameter for the experiment, and `dvc exp run` leaves its result in the workspace: `params.yaml`, `dvc.lock`, the model and `metrics/train.json`. `dvc metrics diff` compares the workspace with the last commit; `--only-changed` makes `dvc exp show` only show the columns that differ from one experiment to another. For `images_per_class`, look at what `train` depends on.

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
git commit -m "Set the learning rate to 0.003"
```

Only `train` runs: `prepare` doesn't read `learning_rate`. In our runs, the accuracy goes from about 0.58 to 0.61 (it varies a little from one machine to another). Changing `images_per_class` would make `prepare` outdated, and `train` would run after it, because its `prepared` dependency would have changed. If `prepare` produced exactly the same files, DVC wouldn't run `train` again: it compares hashes, not dates. `dvc exp show` keeps track of every experiment, even uncommitted: `dvc exp apply` brings one back into the workspace, `dvc exp branch` makes a branch of it.

</details>

## Compressing the model after training

What's left is the `compress` stage, which reduces the trained model and measures what the reduction costs and brings, according to two parameters:

- `sparsity`: the share of the weights of the convolutions and dense layers to set to zero, by a global unstructured pruning of the lowest-magnitude weights, followed by one epoch of retraining; `0` prunes nothing;
- `quant_type`: `none` (no quantization), `dynamic` or `static`, ONNX Runtime's int8 post-training quantizations.

Complete the `compress` function of `src/model_compression/compress.py`, whose beginning provides:

- `export_onnx`, which exports a model to the ONNX format, as a single file, its input named `x`;
- `Calibration`, which gives `quantize_static` images one by one;
- `measure`, which measures the accuracy of an ONNX model on images, the size of its file, raw (`size_mb`) and compressed by gzip (`gzip_mb`), and its median latency for one image (`latency_ms`);
- `load_images`, which gathers the images and labels of a `DataLoader` into arrays.

The function must:

1. load the trained LeNet (`model_path`) and export it to ONNX: that's the reference float model;
2. if `sparsity` is positive, prune the model, retrain it for one epoch with `train_epoch` and an `AdamW` optimizer of rate `learning_rate`, then make the pruning permanent;
3. export the pruned model, then quantize it according to `quant_type` into `onnx_path` (`none`: a plain copy), calibrating the static quantization on a batch of training images;
4. write to `METRICS_PATH`, as JSON, the measures of the float model and of the reduced model on the test images, under the keys `float` and `compressed`.

**Useful documentation**

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
    # Your code here
    ...
```

<details>
<summary>Additional hints</summary>

The exports and quantizations write intermediate files (the float model, the pruned model, the one `quant_pre_process` prepares): a temporary directory keeps them while the function runs. The pruning is the slides', with `amount=sparsity`. `quantize_dynamic` and `quantize_static` read the model prepared by `quant_pre_process`, the latter with `Calibration([image[None] for image in batch])` and `per_channel=True`, `batch` being `train_loader`'s first batch. `load_images(test_loader)` returns the images and labels `measure` expects.

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
            ...  # prune, retrain for one epoch, make the pruning permanent
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

`src/model_compression/compress.py`:

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

## The compression stage

Add to `params.yaml` `sparsity` (`0.5`), `quant_type` (`static`) and `onnx_path` (`landscape_classifier.onnx`), and to the pipeline the `compress` stage, which runs `python -m model_compression compress`, with its parameters, its dependencies, its output and its metric, `metrics/compress.json`. Run the pipeline, compare the float model and the reduced model with `dvc metrics show`, then save everything in git. Is the reduced model smaller? Faster? As accurate?

<details>
<summary>Additional hints</summary>

The parameters of `compress` are the three new ones, plus `prepared_data`, `model_path` and `learning_rate`, which the function reads too. Its dependencies are `prepared`, the trained model and the code it runs: `__main__.py`, `compress.py`, and the modules it imports.

At the end of `params.yaml`:

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

At the end of `params.yaml`:

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
git commit -m "Add the compression stage"
```

In our runs, the model pruned by half then quantized to int8 is three times smaller (0.27 → 0.09 MB), at an accuracy one point lower (0.615 → 0.605)… but slower: 0.7 ms per image instead of 0.45. The quantize and dequantize operations the static quantization adds (the QDQ format) cost more than the int8 computations save on a CPU without fast int8 instructions (VNNI) and on a model this small. A claimed gain isn't a measured gain: measure on the target hardware.

</details>

### Compression experiments

Run two experiments: a dynamic quantization (`quant_type` set to `dynamic`), and a 90% pruning without quantization (`sparsity` set to `0.9`, `quant_type` set to `none`). Compare them with `dvc exp show`. Which setting gives the smallest file? The fastest model? Does pruning alone reduce the file's size? And the compressed file's?

<details>
<summary>Additional hints</summary>

`-S` can be repeated to change several parameters. Only `compress` runs.

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

Both quantizations give files three times smaller, and the dynamic one is the slowest (2.5 ms per image in our runs): it computes the activations' ranges at each inference, and ONNX Runtime's integer convolutions (`ConvInteger`) are slow on a CPU. Pruning alone doesn't reduce the file's size: the zeros are stored like the other weights, and the computations aren't faster. But the compressed file is five times smaller (0.23 → 0.05 MB): zeros compress well, which matters to download a model. And the accuracy doesn't drop: the epoch of retraining makes up for the pruning, a sign that this LeNet is oversized for these images.

</details>

## Solution

The finished project is on the repository's [`solution-en` branch](https://github.com/shuuchuu/compression/tree/solution-en).
