"""Fine-tune CARE ViT-H + the exported Qwen3-VL text decoder on FLARE25."""

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.distributed as dist
import transformers
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer
from transformers.trainer_utils import get_last_checkpoint

from dataset.flare_dataset import FLARECapDataset, validate_split_integrity
from models.llm.qwen3 import VLMQwen3Config, VLMQwen3ForCausalLM


PROJECTOR_STATE_NAME = "mm_projector.safetensors"
TASK_CONFIG_NAME = "task_config.json"
FINAL_CHECKPOINT_DIR = "final_merged"
TASK_NAME = "flare25_task05"
IMAGE_TOKEN = "<im_patch>"
PROJ_TOKENS = 33
FLARE_SYSTEM_PROMPT = "You are an AI assistant trained to act as an abdominal radiologist."
FLARE_USER_PROMPT = " Please describe in detail the findings in this abdomen CT scan.\n        "
EXPECTED_QWEN3 = {
    "hidden_size": 4096,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 12288,
    "vocab_size": 151936,
}


def is_rank_zero():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def rank0_print(*values):
    if is_rank_zero():
        print(*values)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = None
    care_code_root: Optional[str] = None
    care_checkpoint_path: Optional[str] = None
    care_text_assets_path: Optional[str] = None
    input_size: List[int] = field(default_factory=lambda: [112, 336, 336])
    dim: int = 1024
    vision_tower: str = "ours"
    vision_select_layer: int = 0
    vision_select_feature: str = "cls_patch"
    ours_model_name: str = "ViT-H-7-21-21-112-336-336-v3"
    ours_feature_mode: str = "tokens_cls"
    ours_in_channels: int = 1
    mm_projector_type: str = "mlp"
    mm_mlp_depth: int = 2
    mm_projector_norm: bool = True
    proj_out_num: int = PROJ_TOKENS
    attn_implementation: str = "flash_attention_2"


@dataclass
class DataArguments:
    data_root: List[str] = field(default_factory=list)
    json_path: List[str] = field(default_factory=list)
    split_dir: Optional[str] = None
    prompt: str = "simple"


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = "./output/flare25_care_qwen3"
    cache_dir: Optional[str] = None
    model_max_length: int = 1024
    remove_unused_columns: bool = False
    label_names: List[str] = field(default_factory=lambda: ["labels"])
    optim: str = "adamw_torch"
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    num_train_epochs: float = 100.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = False
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 8
    report_to: str = "none"
    lora_enable: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    validate_only: bool = False
    validation_checkpoint: Optional[str] = None
    final_checkpoint_dir: str = FINAL_CHECKPOINT_DIR
    max_shard_size: str = "4GB"


class DataCollator:
    """Original FLARE singular-item keys mapped to the model's plural kwargs."""

    def __call__(self, batch):
        return {
            "images": torch.stack([item["image"] for item in batch]),
            "input_ids": torch.stack([item["input_id"] for item in batch]),
            "labels": torch.stack([item["label"] for item in batch]),
            "attention_mask": torch.stack(
                [item["attention_mask"] for item in batch]
            ),
        }


class SupervisedValidationDataset(Dataset):
    """Add source-equivalent causal-LM labels to deterministic adapter val items."""

    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        prompt = item["question"]
        answer = item["answer"]
        encoded = self.tokenizer(
            prompt + " " + answer,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_id = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]
        valid_len = int(attention_mask.sum().item())
        if valid_len < len(input_id):
            input_id[valid_len] = self.tokenizer.eos_token_id

        prompt_encoded = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        prompt_len = int(prompt_encoded["attention_mask"][0].sum().item())
        label = input_id.clone()
        label[:prompt_len] = -100
        label[label == self.tokenizer.pad_token_id] = -100
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id and valid_len < len(label):
            label[valid_len] = self.tokenizer.eos_token_id
        return {
            "image": item["image"],
            "input_id": input_id,
            "label": label,
            "attention_mask": attention_mask,
        }


def _vlm_from_model(model):
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    while hasattr(model, "module"):
        model = model.module
    if not isinstance(model, VLMQwen3ForCausalLM):
        raise TypeError(f"Expected VLMQwen3ForCausalLM, got {type(model).__name__}")
    return model


def _projector(model):
    return _vlm_from_model(model).get_model().mm_projector


def _save_projector(model, output_dir):
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in _projector(model).state_dict().items()
    }
    save_file(state, str(Path(output_dir) / PROJECTOR_STATE_NAME))


def _load_projector(model, checkpoint):
    path = Path(checkpoint) / PROJECTOR_STATE_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint is missing the trainable projector sidecar: {path}"
        )
    state = load_file(str(path), device="cpu")
    _projector(model).load_state_dict(state, strict=True)
    return str(Path(checkpoint).resolve())


class FLARETrainer(Trainer):
    """HF Trainer plus strict non-LoRA projector checkpoint persistence."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self.best_projector_loaded_from = None

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir=output_dir, state_dict=state_dict)
        output_dir = output_dir or self.args.output_dir
        _save_projector(self.model, output_dir)
        with (Path(output_dir) / TASK_CONFIG_NAME).open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(
                {
                    "schema_version": 1,
                    "task_name": TASK_NAME,
                    "checkpoint_type": "peft_training",
                },
                stream,
                indent=2,
                ensure_ascii=True,
            )
            stream.write("\n")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        _load_projector(model or self.model, resume_from_checkpoint)

    def _load_best_model(self):
        super()._load_best_model()
        self.best_projector_loaded_from = _load_projector(
            self.model, self.state.best_model_checkpoint
        )


def _checkpoint_keys(checkpoint):
    checkpoint = Path(checkpoint)
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as stream:
            index = json.load(stream)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid weight map: {index_path}")
        missing = sorted(
            shard for shard in set(weight_map.values()) if not (checkpoint / shard).is_file()
        )
        if missing:
            raise FileNotFoundError(f"Missing checkpoint shards: {missing}")
        return set(weight_map)
    single = checkpoint / "model.safetensors"
    if single.is_file():
        with safe_open(single, framework="pt", device="cpu") as handle:
            return set(handle.keys())
    raise FileNotFoundError(f"No safetensors model found in {checkpoint}")


def _assert_strict_loading(model, checkpoint, loading_info):
    failures = {
        key: loading_info.get(key, [])
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if loading_info.get(key)
    }
    expected = set(model.state_dict())
    checkpoint_keys = _checkpoint_keys(checkpoint)
    if expected != checkpoint_keys:
        failures["state_key_difference"] = {
            "missing_from_checkpoint": sorted(expected - checkpoint_keys),
            "unexpected_in_checkpoint": sorted(checkpoint_keys - expected),
        }
    if failures:
        raise RuntimeError(f"Decoder checkpoint did not load strictly: {failures}")


def _validate_base_config(config):
    mismatches = {
        name: (getattr(config, name, None), expected)
        for name, expected in EXPECTED_QWEN3.items()
        if getattr(config, name, None) != expected
    }
    if mismatches:
        raise ValueError(f"Unsupported Qwen3-VL text export config: {mismatches}")


def _load_tokenizer(path, max_length):
    kwargs = {
        "local_files_only": True,
        "model_max_length": max_length,
        "padding_side": "right",
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            path, use_fast=False, fix_mistral_regex=True, **kwargs
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, **kwargs)
    tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if len(tokenizer) != 151670 or tokenizer.convert_tokens_to_ids(IMAGE_TOKEN) != 151669:
        raise ValueError(
            "Source-compatible tokenizer resize requires length 151670 and "
            "<im_patch> ID 151669"
        )
    return tokenizer


def _resolve_release_paths(model_args, data_args):
    installed_root = Path(__file__).resolve().parents[3]
    care_root = Path(
        model_args.care_code_root
        or os.environ.get("CARE_CODE_ROOT", installed_root)
    ).expanduser().resolve()
    model_args.care_code_root = str(care_root)
    if model_args.care_checkpoint_path is None:
        model_args.care_checkpoint_path = str(care_root / "ckpts/care.pth")
    else:
        model_args.care_checkpoint_path = str(
            Path(model_args.care_checkpoint_path).expanduser().resolve()
        )
    if data_args.split_dir is None:
        data_args.split_dir = str(
            care_root / "splits/downstream/report_generation/flare25_task05"
        )
    else:
        data_args.split_dir = str(Path(data_args.split_dir).expanduser().resolve())
    return model_args, data_args


def _load_base_model(model_args, training_args, tokenizer):
    if model_args.model_name_or_path is None:
        raise ValueError(
            "--model_name_or_path is required for training or PEFT adapter "
            "validation and must point to the local Qwen3-VL text export"
        )
    os.environ["CARE_CODE_ROOT"] = str(Path(model_args.care_code_root).resolve())
    config = VLMQwen3Config.from_pretrained(
        model_args.model_name_or_path, local_files_only=True
    )
    _validate_base_config(config)
    model, loading_info = VLMQwen3ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        local_files_only=True,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16 if training_args.bf16 else None,
        attn_implementation=model_args.attn_implementation,
        output_loading_info=True,
    )
    _assert_strict_loading(model, model_args.model_name_or_path, loading_info)
    base_vocab_size, resized_vocab_size = model.initialize_vision_tokenizer(
        tokenizer, num_new_tokens=1
    )
    if base_vocab_size != 151936 or resized_vocab_size != 151670:
        raise RuntimeError(
            f"Unexpected decoder resize {base_vocab_size} -> {resized_vocab_size}"
        )
    model.get_model().initialize_vision_modules(model_args)
    model.config.use_cache = False
    model.config.system_prompt = FLARE_SYSTEM_PROMPT
    model.config.user_prompt = FLARE_USER_PROMPT
    model.config.task_name = TASK_NAME
    return model


def _lora_target_modules(model):
    ignored = ("vision_tower", "mm_projector", "embed_tokens", "lm_head")
    return sorted(
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and not any(part in name for part in ignored)
    )


def _prepare_trainable_model(model, training_args):
    if not training_args.lora_enable:
        raise ValueError("FLARE25 release training requires --lora_enable True")
    from peft import LoraConfig, get_peft_model

    model.requires_grad_(False)
    model.enable_input_require_grads()
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    targets = _lora_target_modules(model)
    if not targets:
        raise RuntimeError("No Qwen3 linear modules were selected for LoRA")
    model = get_peft_model(
        model,
        LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
            target_modules=targets,
        ),
    )
    for parameter in _projector(model).parameters():
        parameter.requires_grad = True
    illegal = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name and "mm_projector" not in name
    ]
    if illegal:
        raise RuntimeError(f"Unexpected trainable parameters: {illegal[:20]}")
    return model


def _dataset_args(data_args, model_args, training_args):
    data_args.data_img_size = tuple(model_args.input_size)
    data_args.max_length = training_args.model_max_length
    data_args.proj_out_num = model_args.proj_out_num
    return data_args


def _build_eval_dataset(data_args, tokenizer, training_args):
    base = FLARECapDataset(data_args, tokenizer, mode="validation")
    return SupervisedValidationDataset(base, tokenizer, training_args.model_max_length)


def _resume_checkpoint(training_args):
    output_dir = Path(training_args.output_dir)
    existing = get_last_checkpoint(str(output_dir)) if output_dir.is_dir() else None
    requested = training_args.resume_from_checkpoint
    if requested is None:
        if existing is not None:
            raise ValueError(
                f"Existing checkpoint found at {existing}; pass "
                "--resume_from_checkpoint explicitly or use a new --output_dir"
            )
        return None

    checkpoint = Path(requested).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
    task_path = checkpoint / TASK_CONFIG_NAME
    if not task_path.is_file():
        raise ValueError(f"Resume checkpoint is missing {task_path}")
    with task_path.open(encoding="utf-8") as stream:
        task_config = json.load(stream)
    if task_config.get("task_name") != TASK_NAME:
        raise ValueError(
            f"Resume checkpoint task_name must be {TASK_NAME!r}, got "
            f"{task_config.get('task_name')!r}"
        )
    return str(checkpoint)


def _validate_training_contract(data_args, training_args):
    if data_args.prompt != "simple":
        raise ValueError(
            "FLARE25 training currently requires --prompt simple so training and "
            "merged-checkpoint inference use the same prompt"
        )
    if training_args.eval_strategy == "no" or training_args.save_strategy == "no":
        raise ValueError("Training requires validation and checkpoint saving")
    if training_args.eval_strategy != training_args.save_strategy:
        raise ValueError(
            "--eval_strategy and --save_strategy must match for best-checkpoint loading"
        )
    if not training_args.load_best_model_at_end:
        raise ValueError("Training requires --load_best_model_at_end True")
    if training_args.metric_for_best_model not in ("loss", "eval_loss"):
        raise ValueError("--metric_for_best_model must be loss or eval_loss")
    if training_args.greater_is_better is not False:
        raise ValueError("Validation loss selection requires --greater_is_better False")


def _load_merged_checkpoint(checkpoint, training_args, attn_implementation):
    checkpoint = Path(checkpoint).resolve()
    tokenizer = _load_tokenizer(checkpoint, training_args.model_max_length)
    config = VLMQwen3Config.from_pretrained(checkpoint, local_files_only=True)
    if getattr(config, "task_name", None) != TASK_NAME:
        raise ValueError(
            f"Validation checkpoint task_name must be {TASK_NAME!r}, got "
            f"{getattr(config, 'task_name', None)!r}"
        )
    config.care_checkpoint_path = None
    config.care_text_assets_path = None
    config.ours_resume_path = None
    model, loading_info = VLMQwen3ForCausalLM.from_pretrained(
        checkpoint,
        config=config,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if training_args.bf16 else None,
        attn_implementation=attn_implementation,
        output_loading_info=True,
    )
    _assert_strict_loading(model, checkpoint, loading_info)
    model.requires_grad_(False)
    model.eval()
    return model, tokenizer


def _load_adapter_checkpoint(model_args, training_args, checkpoint):
    from peft import PeftModel

    tokenizer = _load_tokenizer(checkpoint, training_args.model_max_length)
    model = _load_base_model(model_args, training_args, tokenizer)
    model = PeftModel.from_pretrained(
        model, checkpoint, is_trainable=False, local_files_only=True
    )
    _load_projector(model, checkpoint)
    model.requires_grad_(False)
    model.eval()
    return model, tokenizer


def _load_validation_checkpoint(model_args, training_args):
    if not training_args.validation_checkpoint:
        raise ValueError("--validate_only requires --validation_checkpoint")
    checkpoint = Path(training_args.validation_checkpoint).resolve()
    task_path = checkpoint / TASK_CONFIG_NAME
    if not task_path.is_file():
        raise ValueError(
            f"Validation checkpoint is not marked as {TASK_NAME}: missing {task_path}"
        )
    with task_path.open(encoding="utf-8") as stream:
        task_config = json.load(stream)
    if task_config.get("task_name") != TASK_NAME:
        raise ValueError(
            f"Validation checkpoint task_name must be {TASK_NAME!r}, got "
            f"{task_config.get('task_name')!r}"
        )
    if (checkpoint / "adapter_config.json").is_file():
        return _load_adapter_checkpoint(model_args, training_args, checkpoint)
    return _load_merged_checkpoint(
        checkpoint, training_args, model_args.attn_implementation
    )


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_final_merged(trainer, tokenizer, model_args, data_args, training_args, metrics):
    selected = trainer.state.best_model_checkpoint
    if not selected:
        raise RuntimeError("HF Trainer did not select a best validation checkpoint")
    selected = str(Path(selected).resolve())
    if trainer.best_projector_loaded_from != selected:
        raise RuntimeError(
            "Refusing final export: best LoRA was loaded without its matching projector"
        )

    merged = trainer.model.merge_and_unload()
    merged.config.use_cache = True
    merged.config.architectures = [merged.__class__.__name__]
    merged.config.ours_resume_path = None
    merged.config.care_checkpoint_path = None
    merged.config.care_text_assets_path = None
    final_dir = Path(training_args.output_dir) / training_args.final_checkpoint_dir
    if trainer.args.should_save:
        final_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(
            final_dir,
            safe_serialization=True,
            max_shard_size=training_args.max_shard_size,
        )
        tokenizer.save_pretrained(final_dir)
        task_config = {
            "schema_version": 1,
            "task_name": TASK_NAME,
            "task": "FLARE25 Task05 report generation",
            "release_split": "RELEASE_SPLIT_V1",
            "train_csv": str(Path(data_args.split_dir).resolve() / "train.csv"),
            "val_csv": str(Path(data_args.split_dir).resolve() / "val.csv"),
            "train_csv_sha256": _sha256(Path(data_args.split_dir) / "train.csv"),
            "val_csv_sha256": _sha256(Path(data_args.split_dir) / "val.csv"),
            "selected_by": "eval_loss",
            "selected_checkpoint": selected,
            "validation_metrics": metrics,
            "attention_backend": model_args.attn_implementation,
            "base_decoder": {
                "path": str(Path(model_args.model_name_or_path).resolve()),
                "native_vocab_size": 151936,
                "resized_vocab_size": len(tokenizer),
            },
            "vision": {
                "checkpoint": str(Path(model_args.care_checkpoint_path).resolve()),
                "strict_full_care_load": True,
                "feature_tokens": "CLS + adaptive_avg_pool3d(2,4,4)",
                "feature_dimension": 1024,
                "frozen": True,
            },
            "projector": {
                "architecture": "two-layer MLP + LayerNorm",
                "input_dimension": 1024,
                "output_dimension": 4096,
                "tokens": PROJ_TOKENS,
                "initialization": "new for CARE ViT-H + Qwen3; not a historical FLARE checkpoint",
                "checkpoint_selection_reload_verified": True,
            },
            "prompt": {
                "system": FLARE_SYSTEM_PROMPT,
                "user": FLARE_USER_PROMPT,
            },
        }
        with (final_dir / TASK_CONFIG_NAME).open("w", encoding="utf-8") as stream:
            json.dump(task_config, stream, indent=2, ensure_ascii=True)
            stream.write("\n")
    return final_dir


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    transformers.set_seed(training_args.seed)
    model_args, data_args = _resolve_release_paths(model_args, data_args)
    if tuple(model_args.input_size) != (112, 336, 336):
        raise ValueError("CARE ViT-H requires --input_size 112 336 336")
    if model_args.proj_out_num != PROJ_TOKENS:
        raise ValueError("tokens_cls requires --proj_out_num 33")
    if len(data_args.data_root) != 2 or len(data_args.json_path) != 2:
        raise ValueError(
            "Provide two --data_root values and two --json_path values in "
            "CT-AMOS, CT-RATE order"
        )
    validate_split_integrity(data_args.split_dir, include_test=False)
    data_args = _dataset_args(data_args, model_args, training_args)

    if training_args.validate_only:
        model, tokenizer = _load_validation_checkpoint(model_args, training_args)
        eval_dataset = _build_eval_dataset(data_args, tokenizer, training_args)
        trainer = FLARETrainer(
            model=model,
            args=training_args,
            data_collator=DataCollator(),
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        trainer.log_metrics("validation", metrics)
        trainer.save_metrics("validation", metrics)
        return

    _validate_training_contract(data_args, training_args)
    if model_args.model_name_or_path is None:
        raise ValueError(
            "--model_name_or_path is required for training and must point to "
            "the local Qwen3-VL text export"
        )
    tokenizer = _load_tokenizer(
        model_args.model_name_or_path, training_args.model_max_length
    )
    model = _prepare_trainable_model(
        _load_base_model(model_args, training_args, tokenizer), training_args
    )
    train_dataset = FLARECapDataset(data_args, tokenizer, mode="train")
    eval_dataset = _build_eval_dataset(data_args, tokenizer, training_args)
    trainer = FLARETrainer(
        model=model,
        args=training_args,
        data_collator=DataCollator(),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    resume_checkpoint = _resume_checkpoint(training_args)
    if resume_checkpoint:
        rank0_print(f"Resuming from checkpoint: {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_state()
    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    trainer.log_metrics("validation_best", metrics)
    trainer.save_metrics("validation_best", metrics)
    final_dir = _save_final_merged(
        trainer, tokenizer, model_args, data_args, training_args, metrics
    )
    rank0_print(f"Merged best-validation checkpoint: {final_dir}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
