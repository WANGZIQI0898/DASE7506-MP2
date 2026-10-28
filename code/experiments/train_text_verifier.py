"""Train a sparse TF-IDF candidate verifier on training text only."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(CODE_DIR / "experiments") not in sys.path:
    sys.path.insert(0, str(CODE_DIR / "experiments"))

from common import load_inputs, read_jsonl
from train_v2 import evaluate_predictions, unique_answer_counts
from train_verifier import build_labels


def candidate_texts(bank: list[dict], include_question: bool) -> list[str]:
    texts = []
    for row in bank:
        for candidate in row["candidates"]:
            if include_question:
                texts.append(
                    f"{row['question']}\n[SEP]\n{candidate['text']}"
                )
            else:
                texts.append(candidate["text"])
    return texts


def main() -> None:
    tick = time.perf_counter()
    train_bank, _ = load_inputs("train")
    validation_bank, _ = load_inputs("validation")
    train_references = read_jsonl(CODE_DIR / "data" / "gsm8k_train.jsonl")
    validation_references = read_jsonl(CODE_DIR / "data" / "gsm8k_validation.jsonl")
    train_labels = build_labels(train_bank, train_references).reshape(-1)
    validation_labels = build_labels(validation_bank, validation_references)
    validation_answer_counts = unique_answer_counts(validation_bank)

    records = []
    best = None
    for include_question in (False, True):
        view_name = "question_candidate" if include_question else "candidate_only"
        model_tick = time.perf_counter()
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents=None,
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
            min_df=4,
            max_features=30000,
            sublinear_tf=True,
            norm="l2",
            dtype="float32",
        )
        train_matrix = vectorizer.fit_transform(
            candidate_texts(train_bank, include_question)
        )
        validation_matrix = vectorizer.transform(
            candidate_texts(validation_bank, include_question)
        )
        classifier = LogisticRegression(
            C=0.5,
            class_weight="balanced",
            solver="liblinear",
            max_iter=200,
            random_state=20261026,
        )
        classifier.fit(train_matrix, train_labels)
        probabilities = classifier.predict_proba(validation_matrix)[:, 1].reshape(
            len(validation_bank), 4
        )
        choices = probabilities.argmax(axis=1)
        evaluation = evaluate_predictions(
            "validation",
            validation_bank,
            validation_labels,
            choices,
            validation_answer_counts,
        )
        record = {
            "variant": f"tfidf_{view_name}",
            "accuracy": evaluation["accuracy"],
            "correct": evaluation["correct"],
            "non_unanimous_accuracy": evaluation["non_unanimous_accuracy"],
            "accuracy_by_unique_answers": evaluation[
                "accuracy_by_unique_answers"
            ],
            "feature_count": train_matrix.shape[1],
            "training_seconds": round(time.perf_counter() - model_tick, 4),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
        if best is None or (
            record["non_unanimous_accuracy"], record["accuracy"]
        ) > (
            best["record"]["non_unanimous_accuracy"],
            best["record"]["accuracy"],
        ):
            best = {
                "record": record,
                "include_question": include_question,
                "vectorizer": vectorizer,
                "classifier": classifier,
                "probabilities": probabilities,
            }

    model_dir = CODE_DIR / "runs" / "text_verifier_candidate"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "include_question": best["include_question"],
            "vectorizer": best["vectorizer"],
            "classifier": best["classifier"],
        },
        model_dir / "model.joblib",
        compress=3,
    )
    result = {
        "split": "validation",
        "test_used": False,
        "records": records,
        "selected_variant": best["record"]["variant"],
        "selected_validation": best["record"],
        "elapsed_seconds": round(time.perf_counter() - tick, 4),
    }
    output_path = CODE_DIR / "runs" / "text-verifier-development.json"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (model_dir / "config.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "records": records,
                "selected": best["record"]["variant"],
                "model_bytes": (model_dir / "model.joblib").stat().st_size,
                "output": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
