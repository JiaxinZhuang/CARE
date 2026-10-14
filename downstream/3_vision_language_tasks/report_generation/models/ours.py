import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from .llm.qwen3 import VLMQwen3Config, VLMQwen3ForCausalLM


SYSTEM_PROMPT = "You are an AI assistant trained to act as an abdominal radiologist."
USER_PROMPT = " Please describe in detail the findings in this abdomen CT scan.\n        "
PROJ_TOKENS = 33


def build_prompt(
    tokenizer,
    proj_tokens=PROJ_TOKENS,
    system_prompt=SYSTEM_PROMPT,
    user_prompt=USER_PROMPT,
):
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
    return "<im_patch>" * proj_tokens + prompt


def _checkpoint_keys(checkpoint):
    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Merged checkpoint index is required for strict loading: {index_path}"
        )
    with index_path.open(encoding="utf-8") as stream:
        index = json.load(stream)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid checkpoint weight_map: {index_path}")
    missing_shards = sorted(
        shard for shard in set(weight_map.values()) if not (checkpoint / shard).is_file()
    )
    if missing_shards:
        raise FileNotFoundError(f"Checkpoint shards are missing: {missing_shards}")
    return set(weight_map)


def _validate_config(config):
    expected = {
        "task_name": "flare25_task05",
        "model_type": "vlm_qwen3",
        "vision_tower": "ours",
        "ours_model_name": "ViT-H-7-21-21-112-336-336-v3",
        "ours_feature_mode": "tokens_cls",
        "ours_in_channels": 1,
        "vision_select_layer": 0,
        "vision_select_feature": "cls_patch",
        "mm_hidden_size": 1024,
        "mm_projector_type": "mlp",
        "mm_mlp_depth": 2,
        "mm_projector_norm": True,
        "proj_out_num": PROJ_TOKENS,
        "cls_loss_weight": 0.0,
        "cls_num_classes": 8,
        "hidden_size": 4096,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 12288,
        "vocab_size": 151670,
        "eos_token_id": 151645,
    }
    mismatches = {
        name: (getattr(config, name, None), value)
        for name, value in expected.items()
        if getattr(config, name, None) != value
    }
    if tuple(getattr(config, "input_size", ())) != (112, 336, 336):
        mismatches["input_size"] = (
            getattr(config, "input_size", None),
            [112, 336, 336],
        )
    if mismatches:
        details = ", ".join(
            f"{name}={actual!r} (expected {expected_value!r})"
            for name, (actual, expected_value) in sorted(mismatches.items())
        )
        raise ValueError(f"Unsupported merged report checkpoint config: {details}")


def _validate_loading(model, checkpoint_keys, loading_info):
    failures = {
        name: loading_info.get(name, [])
        for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if loading_info.get(name)
    }
    model_keys = set(model.state_dict())
    if model_keys != checkpoint_keys:
        failures["state_key_difference"] = {
            "missing_from_checkpoint": sorted(model_keys - checkpoint_keys),
            "unexpected_in_checkpoint": sorted(checkpoint_keys - model_keys),
        }
    meta_tensors = [name for name, value in model.state_dict().items() if value.is_meta]
    if meta_tensors:
        failures["meta_state_tensors"] = meta_tensors
    if failures:
        raise RuntimeError(
            "Merged checkpoint did not load strictly; refusing partially initialized model: "
            + repr(failures)
        )


class OurWrapper(nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = model.config
        self.input_size = tuple(model.config.input_size)

    @torch.inference_mode()
    def generate(self, images, max_new_tokens=512):
        parameter = next(self.model.parameters())
        images = images.to(parameter.device, dtype=parameter.dtype)
        images = F.interpolate(
            images,
            size=self.input_size,
            mode="trilinear",
            align_corners=False,
        )
        prompt = build_prompt(
            self.tokenizer,
            self.config.proj_out_num,
            getattr(self.config, "system_prompt", SYSTEM_PROMPT),
            getattr(self.config, "user_prompt", USER_PROMPT),
        )
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(
            parameter.device
        )
        output = self.model.generate(
            images=images,
            inputs=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


def load_report_config(checkpoint):
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    config = VLMQwen3Config.from_pretrained(checkpoint, local_files_only=True)
    _validate_config(config)
    # The merged checkpoint owns every visual tensor. Never follow the stale
    # source-training encoder path recorded in the exported config.
    config.ours_resume_path = None
    config.care_checkpoint_path = None
    config.care_text_assets_path = None
    return config


def create_ours(checkpoint, device, config=None):
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(
            "CARE report generation requires a CUDA device and a compatible "
            "FlashAttention 2 installation; no attention backend fallback is supported"
        )
    checkpoint = Path(checkpoint).expanduser().resolve()
    checkpoint_keys = _checkpoint_keys(checkpoint)
    config = config or load_report_config(checkpoint)
    _validate_config(config)
    config.ours_resume_path = None
    config.care_checkpoint_path = None
    config.care_text_assets_path = None

    tokenizer_kwargs = {
        "local_files_only": True,
        "model_max_length": 1024,
        "padding_side": "left",
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, use_fast=False, **tokenizer_kwargs
        )
    except (TypeError, OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, use_fast=True, **tokenizer_kwargs
        )

    if (
        len(tokenizer) != 151670
        or tokenizer.get_vocab().get("<im_patch>") != 151669
        or tokenizer.eos_token_id != 151645
        or tokenizer.pad_token_id != 151643
    ):
        raise ValueError(
            "Tokenizer must have vocabulary length 151670, <im_patch> ID 151669, "
            "EOS ID 151645, and pad ID 151643"
        )
    prompt_ids = tokenizer(build_prompt(tokenizer), return_tensors="pt")["input_ids"][0]
    expected_prefix = [151669] * PROJ_TOKENS + [151644, 8948, 198]
    if prompt_ids[:len(expected_prefix)].tolist() != expected_prefix:
        raise ValueError(
            "Tokenizer must preserve 33 <im_patch> tokens followed by "
            "<|im_start|>system\\n IDs [151644, 8948, 198]"
        )

    model, loading_info = VLMQwen3ForCausalLM.from_pretrained(
        checkpoint,
        config=config,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
        device_map={"": str(device)},
        output_loading_info=True,
    )
    _validate_loading(model, checkpoint_keys, loading_info)

    model.requires_grad_(False)
    model.eval()
    wrapper = OurWrapper(model, tokenizer)
    wrapper.eval()
    return wrapper, tokenizer
