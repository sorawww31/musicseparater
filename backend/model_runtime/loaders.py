"""backend/model_runtime/loaders.py: 固定版コードと検証済み重みからモデルを構築する。"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any


def verify_sha256(path: Path, expected: str) -> None:
    """破損または差し替えられた checkpoint をロード前に拒否する。"""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise RuntimeError(f"checkpoint の SHA-256 が一致しません: {path.name}")


def _install_asteroid_compatibility(torch: Any) -> None:
    """UNMIXX が使う Asteroid の単純な長さ合わせだけを提供する。"""
    asteroid = types.ModuleType("asteroid")
    utils = types.ModuleType("asteroid.utils")
    torch_utils = types.ModuleType("asteroid.utils.torch_utils")

    def pad_x_to_y(x: Any, y: Any) -> Any:
        if x.shape[-1] >= y.shape[-1]:
            return x[..., : y.shape[-1]]
        return torch.nn.functional.pad(x, (0, y.shape[-1] - x.shape[-1]))

    torch_utils.pad_x_to_y = pad_x_to_y
    sys.modules.update({
        "asteroid": asteroid,
        "asteroid.utils": utils,
        "asteroid.utils.torch_utils": torch_utils,
    })


def _load_unmixx_class(source_directory: Path, torch: Any) -> type[Any]:
    """重い任意依存を含む upstream package __init__ を通さず必要ファイルだけ読む。"""
    _install_asteroid_compatibility(torch)
    package_paths = {
        "look2hear": source_directory / "look2hear",
        "look2hear.models": source_directory / "look2hear" / "models",
        "look2hear.models.layers": source_directory / "look2hear" / "models" / "layers",
    }
    for name, path in package_paths.items():
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package

    def load(name: str, path: Path) -> Any:
        specification = importlib.util.spec_from_file_location(name, path)
        if specification is None or specification.loader is None:
            raise RuntimeError(f"UNMIXX module を読み込めません: {path}")
        module = importlib.util.module_from_spec(specification)
        sys.modules[name] = module
        specification.loader.exec_module(module)
        return module

    models = package_paths["look2hear.models"]
    layers = package_paths["look2hear.models.layers"]
    load("look2hear.models.base_model", models / "base_model.py")
    layers_package = sys.modules["look2hear.models.layers"]
    layers_package.activations = load(
        "look2hear.models.layers.activations", layers / "activations.py",
    )
    layers_package.normalizations = load(
        "look2hear.models.layers.normalizations", layers / "normalizations.py",
    )
    module = load("look2hear.models.unmixx_model", models / "unmixx_model.py")
    return module.UNMIXX


def load_unmixx(settings: dict[str, Any], device: Any) -> Any:
    """公式 UNMIXX コードと同梱 checkpoint を strict にロードする。"""
    import torch
    import yaml

    source_directory = Path(settings["source_directory"])
    checkpoint = source_directory / settings["checkpoint_filename"]
    configuration = source_directory / "ckpt" / "conf.yml"
    if not source_directory.is_dir() or not configuration.is_file():
        raise RuntimeError("UNMIXX の固定版ソースが Docker image にありません")
    verify_sha256(checkpoint, settings["checkpoint_sha256"])
    UNMIXX = _load_unmixx_class(source_directory, torch)

    with configuration.open(encoding="utf-8") as stream:
        configuration_data = yaml.safe_load(stream)
    arguments = dict(configuration_data["audionet"]["audionet_config"])
    # upstream の build_model と同じく、周波数は audionet_config 外から明示する。
    arguments["sample_rate"] = configuration_data["datamodule"]["data_config"]["sample_rate"]
    # upstream constructor の周波数帯一覧 print は子プロセスの進捗プロトコルに不要。
    with redirect_stdout(io.StringIO()):
        model = UNMIXX(**arguments)
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw)
    state = {
        key.removeprefix("audio_model."): value
        for key, value in state.items()
        if key.startswith("audio_model.")
    }
    model.load_state_dict(state, strict=True)
    return model.to(device=device, dtype=torch.float32).eval()


def load_sepacap(settings: dict[str, Any], device: Any) -> Any:
    """Hugging Face から SepACap の設定と重みを固定 revision で取得する。"""
    import torch
    import yaml
    from huggingface_hub import hf_hub_download
    from loguru import logger

    cache_dir = settings.get("cache_dir")
    common = {"repo_id": settings["repo_id"], "revision": settings["revision"], "cache_dir": cache_dir}
    checkpoint = Path(hf_hub_download(filename=settings["checkpoint_filename"], **common))
    configuration = Path(hf_hub_download(filename=settings["config_filename"], **common))
    verify_sha256(checkpoint, settings["checkpoint_sha256"])

    source_directory = Path(settings["source_directory"])
    if not source_directory.is_dir():
        raise RuntimeError("SepACap の固定版ソースが Docker image にありません")
    # upstream の class decorator は生成した巨大なモデル全体を DEBUG 出力する。
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    sys.path.insert(0, str(source_directory))
    from src.model import Model

    with configuration.open(encoding="utf-8") as stream:
        arguments = yaml.safe_load(stream)["config"]["model"]
    model = Model(**arguments)
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = raw.get("model_state", raw.get("state_dict", raw))
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device=device, dtype=torch.bfloat16).eval()


def load_singer_informed(settings: dict[str, Any], device: Any) -> tuple[Any, Any]:
    """最良のConcatenation λ=0.1モデルとclean embeddingを厳密にロードする。"""
    import torch

    from .singer_informed import build_models

    checkpoint = Path(settings["checkpoint_path"])
    embedding_checkpoint = Path(settings["embedding_checkpoint_path"])
    verify_sha256(checkpoint, settings["checkpoint_sha256"])
    verify_sha256(embedding_checkpoint, settings["embedding_checkpoint_sha256"])

    separator, embedding_model = build_models(settings, torch)
    separator_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    embedding_state = torch.load(
        embedding_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    separator.load_state_dict(separator_state, strict=True)
    embedding_model.load_state_dict(embedding_state, strict=True)
    return (
        separator.to(device=device, dtype=torch.float32).eval(),
        embedding_model.to(device=device, dtype=torch.float32).eval(),
    )
