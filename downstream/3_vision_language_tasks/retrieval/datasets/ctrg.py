import csv
import json
from pathlib import Path
from typing import Any


def _pair_id(value: Any) -> str:
    if value is None:
        return ""
    name = Path(str(value).strip()).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def load_ctrg_rows(
    manifest_path: str | Path,
    data_root: str | Path,
    *,
    validate_images: bool = True,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    manifest = Path(manifest_path).expanduser().resolve()
    root = Path(data_root).expanduser().absolute()
    annotations = root / "annotations.json"
    if not annotations.is_file():
        raise ValueError(f"CTRG annotations do not exist: {annotations}")

    split_paths: dict[str, str] = {}
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or []) != {"pair_id", "image_path"}:
            raise ValueError("CTRG manifest must contain exactly pair_id,image_path")
        for line_number, raw in enumerate(reader, start=2):
            pair_id = _pair_id(raw.get("pair_id"))
            image_path = str(raw.get("image_path") or "").strip()
            if not pair_id or not image_path:
                raise ValueError(f"CTRG manifest line {line_number} has empty fields")
            if Path(image_path).is_absolute():
                raise ValueError(f"CTRG manifest line {line_number} image_path must be relative")
            if pair_id in split_paths:
                raise ValueError(f"Duplicate CTRG pair_id: {pair_id}")
            split_paths[pair_id] = image_path
    if not split_paths:
        raise ValueError("CTRG manifest contains no pairs")

    payload = json.loads(annotations.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(
        isinstance(payload.get(name), list) for name in ("train", "test")
    ):
        raise ValueError("CTRG annotations must contain train and test lists")

    annotation_rows: list[tuple[str, dict[str, Any]]] = []
    annotation_ids: set[str] = set()
    for split_name in ("train", "test"):
        for raw in payload[split_name]:
            if not isinstance(raw, dict):
                raise ValueError(f"CTRG {split_name} annotations must contain objects")
            pair_id = _pair_id(raw.get("id"))
            if not pair_id:
                raise ValueError(f"CTRG {split_name} annotation has an empty id")
            if pair_id in annotation_ids:
                raise ValueError(f"Duplicate CTRG annotation id: {pair_id}")
            annotation_ids.add(pair_id)
            annotation_rows.append((pair_id, raw))

    split_ids = set(split_paths)
    if split_ids != annotation_ids:
        missing = sorted(split_ids - annotation_ids)
        unexpected = sorted(annotation_ids - split_ids)
        raise ValueError(
            f"CTRG split/annotation IDs differ: missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )
    annotation_order = [pair_id for pair_id, _ in annotation_rows]
    if list(split_paths) != annotation_order:
        raise ValueError(
            "CTRG manifest order must match train-then-test annotation order"
        )

    rows: list[dict[str, str]] = []
    seen_texts: set[str] = set()
    dropped_ids: list[str] = []
    for pair_id, raw in annotation_rows:
        text = f"{raw.get('report', '')} {raw.get('result', '')}".strip()
        if not text:
            raise ValueError(f"CTRG annotation {pair_id} has no report text")
        if text in seen_texts:
            dropped_ids.append(pair_id)
            continue
        seen_texts.add(text)

        image_path = root / split_paths[pair_id]
        if validate_images and not image_path.is_file():
            raise ValueError(f"CTRG image does not exist: {image_path}")
        rows.append({
            "pair_id": pair_id,
            "image_path": str(image_path),
            "text_en": text,
            "report_key": pair_id,
            "body_part": "chest",
        })
    if not rows:
        raise ValueError("No valid CTRG pairs remain after exact-report deduplication")

    deduplication = {
        "policy": "keep first exact combined report+result in train,test annotation order",
        "dropped_report_count": len(dropped_ids),
        "dropped_report_ids": dropped_ids,
    }
    source = {
        "profile": "ctrg",
        "data_root": str(root),
        "annotations_path": str(annotations),
        "split_pair_count": len(split_paths),
        "annotation_order": ["train", "test"],
        "manifest_order": "train,test annotation order",
        "patient_grouping": "UNKNOWN",
    }
    return rows, deduplication, source
