import argparse
import csv
import json
from pathlib import Path

import numpy as np
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


PREDICTION_COLUMN = "generated_report"
REFERENCE_COLUMN = "reference_report"


def _tokenize(text, row_number, column):
    try:
        return word_tokenize(text.lower())
    except LookupError as error:
        raise RuntimeError(
            "Required NLTK tokenizer data is unavailable while reading "
            f"row {row_number}, column {column!r}. Install the required NLTK "
            "resources outside this script; automatic downloads are disabled."
        ) from error


def load_report_pairs(prediction_csv):
    predictions = []
    references = []
    with Path(prediction_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = {PREDICTION_COLUMN, REFERENCE_COLUMN} - fieldnames
        if missing:
            raise ValueError(f"Prediction CSV is missing columns: {sorted(missing)}")

        for row_number, row in enumerate(reader, start=2):
            prediction = row[PREDICTION_COLUMN]
            reference = row[REFERENCE_COLUMN]
            if prediction is None or reference is None:
                raise ValueError(f"Malformed prediction CSV row {row_number}")
            predictions.append(prediction)
            references.append(reference)

    if not predictions:
        raise ValueError("Prediction CSV contains no report rows")
    return predictions, references


def evaluate_reports(predictions, references):
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        raise ValueError("At least one prediction/reference pair is required")
    for row_number, reference in enumerate(references, start=2):
        if not reference.strip():
            raise ValueError(
                f"Blank reference_report at row {row_number}; "
                "ground truth is required for evaluation"
            )

    smoothing = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    bleu1_scores = []
    bleu4_scores = []
    rouge_l_scores = []
    meteor_scores = []

    for row_number, (prediction, reference) in enumerate(
        zip(predictions, references), start=2
    ):
        prediction_tokens = _tokenize(prediction, row_number, PREDICTION_COLUMN)
        reference_tokens = _tokenize(reference, row_number, REFERENCE_COLUMN)

        bleu1_scores.append(
            sentence_bleu(
                [reference_tokens],
                prediction_tokens,
                weights=(1.0, 0.0, 0.0, 0.0),
                smoothing_function=smoothing,
            )
        )
        bleu4_scores.append(
            sentence_bleu(
                [reference_tokens],
                prediction_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothing,
            )
        )
        rouge_l_scores.append(rouge.score(reference, prediction)["rougeL"].fmeasure)
        try:
            meteor_scores.append(meteor_score([reference_tokens], prediction_tokens))
        except LookupError as error:
            raise RuntimeError(
                "Required NLTK METEOR data is unavailable while evaluating row "
                f"{row_number}. Install the required NLTK resources outside this "
                "script; automatic downloads are disabled."
            ) from error

    return {
        "BLEU1": float(np.mean(bleu1_scores)),
        "BLEU4": float(np.mean(bleu4_scores)),
        "ROUGEL": float(np.mean(rouge_l_scores)),
        "METEOR": float(np.mean(meteor_scores)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated radiology reports")
    parser.add_argument("--prediction-csv", required=True, help="CSV with generated_report and reference_report columns")
    parser.add_argument("--output-json", required=True, help="Path for the metrics JSON")
    args = parser.parse_args()

    predictions, references = load_report_pairs(args.prediction_csv)
    metrics = evaluate_reports(predictions, references)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
