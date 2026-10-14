import csv
import json
import os
from pathlib import Path

import monai.transforms as mtf
import numpy as np
import SimpleITK as sitk
import torch
from monai.data import set_track_meta
from skimage.transform import resize
from torch.utils.data import Dataset


SOURCE_DATASETS = ("CT-AMOS", "CT-RATE")
REGISTRY_FIELDS = ["case_id", "source_dataset", "care_split"]
EXPECTED_SPLITS = {
    "train": {"rows": 2589, "sources": {"CT-AMOS": 989, "CT-RATE": 1600}},
    "val": {"rows": 647, "sources": {"CT-AMOS": 247, "CT-RATE": 400}},
    "test": {"rows": 681, "sources": {"CT-AMOS": 387, "CT-RATE": 294}},
}
PREPROCESS_SIZE = (128, 256, 256)
WINDOWS = {"CT-AMOS": (-160.0, 240.0), "CT-RATE": (-1350.0, 150.0)}

prompt_templates = {
    "simple": (
        "You are an AI assistant trained to act as an abdominal radiologist. "
        "Please describe in detail the findings in this abdomen CT scan.\n        "
    )
}


def _default_split_dir():
    return (
        Path(__file__).resolve().parents[4]
        / "splits/downstream/report_generation/flare25_task05"
    )


def _read_registry(split_dir, split):
    path = Path(split_dir) / f"{split}.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != REGISTRY_FIELDS:
            raise ValueError(
                f"{path} must have exactly the columns {REGISTRY_FIELDS}, "
                f"got {reader.fieldnames}"
            )
        rows = list(reader)

    expected = EXPECTED_SPLITS[split]
    if len(rows) != expected["rows"]:
        raise ValueError(
            f"{path} has {len(rows)} rows; RELEASE_SPLIT_V1 requires "
            f"{expected['rows']}"
        )

    ids = []
    source_counts = {source: 0 for source in SOURCE_DATASETS}
    for row_number, row in enumerate(rows, start=2):
        case_id = row["case_id"]
        source = row["source_dataset"]
        if not case_id:
            raise ValueError(f"{path}:{row_number} has an empty case_id")
        if source not in source_counts:
            raise ValueError(
                f"{path}:{row_number} has unsupported source_dataset {source!r}"
            )
        if row["care_split"] != split:
            raise ValueError(
                f"{path}:{row_number} has care_split={row['care_split']!r}; "
                f"expected {split!r}"
            )
        ids.append(case_id)
        source_counts[source] += 1

    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate case_id values")
    if source_counts != expected["sources"]:
        raise ValueError(
            f"{path} source counts are {source_counts}; RELEASE_SPLIT_V1 requires "
            f"{expected['sources']}"
        )
    return rows


def validate_split_integrity(split_dir=None, include_test=False):
    """Validate RELEASE_SPLIT_V1 registries without opening annotation JSON."""
    split_dir = Path(split_dir) if split_dir is not None else _default_split_dir()
    rows = {
        "train": _read_registry(split_dir, "train"),
        "val": _read_registry(split_dir, "val"),
    }
    train_ids = {row["case_id"] for row in rows["train"]}
    val_ids = {row["case_id"] for row in rows["val"]}
    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(f"RELEASE_SPLIT_V1 train/val overlap contains {len(overlap)} IDs")

    if include_test:
        rows["test"] = _read_registry(split_dir, "test")
        test_ids = {row["case_id"] for row in rows["test"]}
        if train_ids & test_ids or val_ids & test_ids:
            raise ValueError("RELEASE_SPLIT_V1 development/test case_id overlap detected")
    return {split: len(split_rows) for split, split_rows in rows.items()}


def _load_json(path):
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, list):
        raise ValueError(f"Annotation JSON must contain a list: {path}")
    return document


def _validate_annotation(item, path, index):
    if not isinstance(item, dict):
        raise ValueError(f"{path}: item {index} must be an object")
    missing = {"case_id", "findings"} - item.keys()
    if missing:
        raise ValueError(f"{path}: item {index} is missing {sorted(missing)}")
    if not isinstance(item["case_id"], str) or not item["case_id"]:
        raise ValueError(f"{path}: item {index} has an invalid case_id")
    if not isinstance(item["findings"], dict) or not item["findings"]:
        raise ValueError(f"{path}: item {index} has invalid findings")
    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in item["findings"].items()
    ):
        raise ValueError(f"{path}: item {index} findings must map strings to strings")


def _format_report(item):
    return "\n".join(
        f"{key}: {value}" for key, value in item["findings"].items()
    )


def read_image(path, source_dataset):
    path = Path(path)
    if path.suffix == ".npy":
        image = np.load(path, allow_pickle=False)
    else:
        itk_image = sitk.ReadImage(str(path))
        volume = sitk.GetArrayFromImage(itk_image)
        if volume.ndim == 4:
            volume = volume[0]
        if volume.ndim != 3:
            raise ValueError(f"Expected a 3D volume, got {volume.shape} from {path}")
        lower, upper = WINDOWS[source_dataset]
        volume = np.clip(volume, lower, upper)
        minimum = float(volume.min())
        maximum = float(volume.max())
        if maximum == minimum:
            volume = np.zeros_like(volume, dtype=np.float32)
        else:
            volume = (volume - minimum) / (maximum - minimum)
        volume = resize(volume, PREPROCESS_SIZE, anti_aliasing=True)
        image = volume.astype(np.float32, copy=False)[None]

    image = np.asarray(image)
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"Expected image shape [1,D,H,W], got {image.shape} from {path}")
    return image


class FLARECapDataset(Dataset):
    """FLARE25 report dataset backed by RELEASE_SPLIT_V1 CSV registries.

    ``args.json_path`` is ``[CT-AMOS JSON, CT-RATE JSON]`` for train and
    validation, and ``[official validation/val.json]`` for test.
    ``args.data_root`` is always ``[CT-AMOS root, CT-RATE root]``.
    """

    def __init__(self, args, tokenizer, mode="train"):
        if mode not in {"train", "validation", "test"}:
            raise ValueError("mode must be 'train', 'validation', or 'test'")

        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.image_tokens = "<im_patch>" * args.proj_out_num
        self.split_dir = Path(getattr(args, "split_dir", _default_split_dir()))
        validate_split_integrity(self.split_dir, include_test=False)

        roots = list(args.data_root) if isinstance(args.data_root, (list, tuple)) else []
        if len(roots) != 2:
            raise ValueError(
                "args.data_root must be [CT-AMOS root, CT-RATE root]"
            )
        self.data_roots = dict(zip(SOURCE_DATASETS, map(Path, roots)))

        json_paths = (
            list(args.json_path)
            if isinstance(args.json_path, (list, tuple))
            else [args.json_path]
        )
        registry_split = "val" if mode == "validation" else mode
        registry = _read_registry(self.split_dir, registry_split)
        self.data_list = self._join_annotations(json_paths, registry)

        prompt = prompt_templates.get(args.prompt, args.prompt)
        self.system = prompt.split(".", 1)[0] + "."
        self.content = prompt.split(".", 1)[1] if "." in prompt else ""
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.content},
        ]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False)
        self.prompt = self.image_tokens + rendered

        train_transform = mtf.Compose(
            [
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                mtf.ToTensor(dtype=torch.float),
            ]
        )
        deterministic_transform = mtf.Compose([mtf.ToTensor(dtype=torch.float)])
        self.transform = train_transform if mode == "train" else deterministic_transform
        self.resize_transform = mtf.Resize(args.data_img_size)
        set_track_meta(False)

    def _join_annotations(self, json_paths, registry):
        annotations = {}
        if self.mode in {"train", "validation"}:
            if len(json_paths) != 2:
                raise ValueError(
                    "train/validation args.json_path must be "
                    "[CT-AMOS JSON, CT-RATE JSON]"
                )
            for source, json_path in zip(SOURCE_DATASETS, json_paths):
                for index, item in enumerate(_load_json(json_path)):
                    _validate_annotation(item, json_path, index)
                    key = (source, item["case_id"])
                    if key in annotations:
                        raise ValueError(
                            f"Duplicate annotation case_id for {source}: {item['case_id']}"
                        )
                    annotations[key] = item
        else:
            if len(json_paths) != 1:
                raise ValueError("test args.json_path must be [validation/val.json]")
            json_path = json_paths[0]
            by_id = {}
            for index, item in enumerate(_load_json(json_path)):
                _validate_annotation(item, json_path, index)
                case_id = item["case_id"]
                if case_id in by_id:
                    raise ValueError(f"Duplicate test annotation case_id: {case_id}")
                by_id[case_id] = item
            annotations = {
                (row["source_dataset"], row["case_id"]): by_id[row["case_id"]]
                for row in registry
                if row["case_id"] in by_id
            }

        records = []
        missing_annotations = []
        missing_volumes = []
        for row in registry:
            key = (row["source_dataset"], row["case_id"])
            item = annotations.get(key)
            if item is None:
                missing_annotations.append(row["case_id"])
                continue
            image_path = self.data_roots[row["source_dataset"]] / row["case_id"]
            if not image_path.is_file():
                missing_volumes.append(str(image_path))
                continue
            records.append(
                {
                    "sample_id": row["case_id"],
                    "source_dataset": row["source_dataset"],
                    "image_path": str(image_path),
                    "report": _format_report(item),
                }
            )

        if missing_annotations:
            raise ValueError(
                f"Exact registry/annotation join failed for {len(missing_annotations)} cases; "
                f"first missing ID: {missing_annotations[0]}"
            )
        if missing_volumes:
            raise FileNotFoundError(
                f"Exact registry/root join failed for {len(missing_volumes)} volumes; "
                f"first missing path: {missing_volumes[0]}"
            )
        if len(records) != len(registry):
            raise RuntimeError("FLARE25 exact join did not preserve every registry row")
        return records

    def __len__(self):
        return len(self.data_list)

    def _load_image(self, record):
        image = read_image(record["image_path"], record["source_dataset"])
        image = self.transform(image)
        return self.resize_transform(image)

    def _inference_item(self, record, image):
        input_id = self.tokenizer(self.prompt, return_tensors="pt")["input_ids"]
        return {
            "image": image,
            "input_id": input_id,
            "prompt": self.prompt,
            "report": record["report"],
            "sample_id": record["sample_id"],
            "image_path": record["image_path"],
            "question": self.prompt,
            "answer": record["report"],
            "image_name": record["sample_id"],
            "question_type": "Caption",
        }

    def _training_item(self, record, image):
        text = self.tokenizer(
            self.prompt + " " + record["report"],
            max_length=self.args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_id = text["input_ids"][0]
        attention_mask = text["attention_mask"][0]
        valid_len = int(attention_mask.sum().item())
        if valid_len < len(input_id):
            input_id[valid_len] = self.tokenizer.eos_token_id

        prompt_tokens = self.tokenizer(
            self.prompt,
            max_length=self.args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        prompt_len = int(prompt_tokens["attention_mask"][0].sum().item())
        label = input_id.clone()
        label[:prompt_len] = -100
        label[label == self.tokenizer.pad_token_id] = -100
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id and valid_len < len(label):
            label[valid_len] = self.tokenizer.eos_token_id

        return {
            "image": image,
            "input_id": input_id,
            "label": label,
            "attention_mask": attention_mask,
            "question": self.prompt,
            "answer": record["report"],
            "sample_id": record["sample_id"],
            "question_type": "Caption",
        }

    def __getitem__(self, index):
        record = self.data_list[index]
        image = self._load_image(record)
        if self.mode == "train":
            return self._training_item(record, image)
        return self._inference_item(record, image)
