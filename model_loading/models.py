import importlib
import inspect
import json
from collections.abc import Mapping
from pathlib import Path

import torch

if __package__:
    from .ct_transform import get_transform
else:
    from ct_transform import get_transform

__all__ = ["get_model", "get_transform", "get_tokenizer"]

_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "care_vit_h.json"
_DEFAULT_TEXT_ASSETS = Path(__file__).resolve().parent / "text_assets"
_OPEN_CLIP_PATH = Path(__file__).resolve().parent / "open_clip"
_PREREQUISITE = (
    "CARE requires the bundled custom 3D OpenCLIP package, including "
    "CustomTextCLIP with use_fine_grained support; upstream 2D open_clip "
    "is not sufficient."
)


def _open_clip():
    module_name = f"{__package__}.open_clip" if __package__ else "open_clip"
    try:
        package = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(_PREREQUISITE) from exc
    package_file = getattr(package, "__file__", None)
    if package_file is None or Path(package_file).resolve().parent != _OPEN_CLIP_PATH:
        raise ImportError(
            f"{_PREREQUISITE} Resolved {module_name!r} outside the bundled package: "
            f"{package_file!r}."
        )
    required = ("create_model", "add_model_config", "get_model_config", "get_tokenizer")
    constructor = getattr(package, "CustomTextCLIP", None)
    if (
        any(not callable(getattr(package, name, None)) for name in required)
        or constructor is None
        or "use_fine_grained" not in inspect.signature(constructor).parameters
    ):
        raise ImportError(_PREREQUISITE)
    return package


def _text_assets(text_assets_path):
    path = Path(text_assets_path) if text_assets_path is not None else _DEFAULT_TEXT_ASSETS
    path = path.expanduser().resolve()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(
            f"Local BERT text assets (including config.json) not found: {path}. "
            "Provide text_assets_path or populate the sibling text_assets directory; "
            "assets are not downloaded automatically."
        )
    # Prevent local assets from fetching remote code.
    for name in ("config.json", "tokenizer_config.json"):
        config_path = path / name
        if config_path.is_file():
            with config_path.open(encoding="utf-8") as handle:
                config = json.load(handle)
            for reference in config.get("auto_map", {}).values():
                references = reference if isinstance(reference, (list, tuple)) else [reference]
                if any(isinstance(item, str) and "--" in item for item in references):
                    raise ValueError(
                        f"{config_path} references remote code. Provide self-contained "
                        "local text assets with local auto_map references; no Hub downloads."
                    )
    return path


def _checkpoint_state(checkpoint_path):
    path = Path(checkpoint_path)
    if path.suffix not in (".pt", ".pth"):
        raise ValueError("CARE checkpoints must be .pt or .pth files.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Expected a flat tensor state dict or a checkpoint with state_dict.")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint state_dict must be a nonempty flat tensor mapping.")
    normalized = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("Checkpoint state_dict must be a flat string-to-tensor mapping.")
        key = key.removeprefix("module.")
        if key in normalized:
            raise ValueError(f"Duplicate checkpoint key after removing module. prefix: {key}")
        normalized[key] = value
    return normalized


def get_model(device, checkpoint_path, *, text_assets_path=None):
    """Load a full joint CARE checkpoint strictly and return an FP32 eval model."""
    assets = _text_assets(text_assets_path)
    package = _open_clip()
    state = _checkpoint_state(checkpoint_path)
    package.add_model_config(_CONFIG_PATH)
    config = package.get_model_config(_CONFIG_PATH.stem)
    text_cfg = dict(config["text_cfg"])
    text_cfg.update(
        hf_model_name=str(assets),
        hf_tokenizer_name=str(assets),
        hf_model_pretrained=False,
    )
    model = package.create_model(
        _CONFIG_PATH.stem,
        pretrained=None,
        load_weights=False,
        pretrained_image=False,
        pretrained_text=False,
        precision="fp32",
        device="cpu",
        text_cfg=text_cfg,
        use_fine_grained=True,
    )
    model.load_state_dict(state, strict=True)
    return model.float().to(device).eval()


def get_tokenizer(*, text_assets_path=None):
    """Return the bundled local BERT tokenizer with context length 512."""
    assets = _text_assets(text_assets_path)
    package = _open_clip()
    tokenizer_module = importlib.import_module(f"{package.__name__}.tokenizer")

    return tokenizer_module.HFTokenizer(
        str(assets),
        context_length=512,
        clean="whitespace",
        local_files_only=True,
    )
