#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from firac_appeals.utils.io import read_jsonl, write_jsonl


LABELS = ["allowed", "dismissed"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


class OutcomeDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        text = str(row.get("input", {}).get("input_text", ""))
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoded["labels"] = LABEL_TO_ID[str(row["target"]["label"]).lower()]
        encoded["case_id"] = str(row["case_id"])
        return encoded


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a legal encoder for Facts-only outcome prediction.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--model", default="law-ai/InLegalBERT")
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260604, 20260605, 20260606])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required")

    train_rows = valid_rows(read_jsonl(args.train))
    test_rows = valid_rows(read_jsonl(args.test))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_results = []

    for seed in args.seeds:
        result = run_seed(args, train_rows, test_rows, tokenizer, seed, output_dir)
        seed_results.append(result)

    metric_names = ["accuracy", "macro_f1", "mcc"]
    summary = {
        "dataset": args.dataset_name,
        "model": args.model,
        "train": args.train,
        "test": args.test,
        "train_count": len(train_rows),
        "test_count": len(test_rows),
        "seeds": args.seeds,
        "per_seed": seed_results,
        "aggregate": {
            metric: {
                "mean": mean(float(item[metric]) for item in seed_results),
                "std": stdev(float(item[metric]) for item in seed_results) if len(seed_results) > 1 else 0.0,
            }
            for metric in metric_names
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))


def run_seed(
    args: argparse.Namespace,
    all_train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    tokenizer: Any,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    set_seed(seed)
    labels = [str(row["target"]["label"]).lower() for row in all_train_rows]
    train_rows, validation_rows = train_test_split(
        all_train_rows,
        test_size=args.validation_size,
        random_state=seed,
        stratify=labels,
    )
    class_counts = Counter(str(row["target"]["label"]).lower() for row in train_rows)
    weights = torch.tensor(
        [len(train_rows) / (len(LABELS) * class_counts[label]) for label in LABELS],
        dtype=torch.float32,
        device="cuda",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id=LABEL_TO_ID,
    ).to("cuda")
    collator = OutcomeCollator(tokenizer)
    train_loader = make_loader(train_rows, tokenizer, collator, args, shuffle=True)
    validation_loader = make_loader(validation_rows, tokenizer, collator, args, shuffle=False)
    test_loader = make_loader(test_rows, tokenizer, collator, args, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    loss_function = torch.nn.CrossEntropyLoss(weight=weights)
    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            case_ids = batch.pop("case_id")
            del case_ids
            labels_tensor = batch.pop("labels").to("cuda")
            inputs = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**inputs).logits
                loss = loss_function(logits, labels_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))

        validation = predict(model, validation_loader)
        history.append(
            {
                "epoch": epoch,
                "train_loss": mean(losses),
                "validation_macro_f1": validation["macro_f1"],
            }
        )
        print(
            f"{args.dataset_name} seed={seed} epoch={epoch} "
            f"loss={mean(losses):.4f} val_macro_f1={validation['macro_f1']:.4f}",
            flush=True,
        )
        if validation["macro_f1"] > best_score:
            best_score = float(validation["macro_f1"])
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience and epochs_without_improvement >= args.early_stopping_patience:
                break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    test_result = predict(model, test_loader)
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(seed_dir / "checkpoint")
    tokenizer.save_pretrained(seed_dir / "checkpoint")
    predictions = [
        {
            "case_id": case_id,
            "label": LABELS[predicted],
            "model": args.model,
            "prediction_status": "ok",
            "probabilities": {LABELS[index]: probability for index, probability in enumerate(probs)},
            "task": "outcome_prediction",
        }
        for case_id, predicted, probs in zip(
            test_result.pop("case_ids"),
            test_result.pop("predicted_ids"),
            test_result.pop("probabilities"),
        )
    ]
    write_jsonl(seed_dir / "predictions.jsonl", predictions)
    per_case = [
        {"case_id": row["case_id"], "gold": gold, "predicted": prediction["label"]}
        for row, gold, prediction in zip(
            test_rows,
            [str(row["target"]["label"]).lower() for row in test_rows],
            predictions,
        )
    ]
    write_jsonl(seed_dir / "per_case_metrics.jsonl", per_case)
    metrics = {
        "task": "outcome_prediction",
        "count": len(test_rows),
        "aggregate": test_result,
        "references": args.test,
        "predictions": str(seed_dir / "predictions.jsonl"),
        "model": args.model,
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_macro_f1": best_score,
        "history": history,
    }
    (seed_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    del model, optimizer, scheduler, best_state
    torch.cuda.empty_cache()
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_macro_f1": best_score,
        **test_result,
        "metrics": str(seed_dir / "metrics.json"),
    }


class OutcomeCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        case_ids = [row.pop("case_id") for row in rows]
        labels = torch.tensor([row.pop("labels") for row in rows], dtype=torch.long)
        batch = self.tokenizer.pad(rows, padding=True, return_tensors="pt")
        batch["labels"] = labels
        batch["case_id"] = case_ids
        return batch


def make_loader(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    collator: OutcomeCollator,
    args: argparse.Namespace,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        OutcomeDataset(rows, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )


@torch.inference_mode()
def predict(model: Any, loader: DataLoader) -> dict[str, Any]:
    model.eval()
    case_ids = []
    gold_ids = []
    predicted_ids = []
    probabilities = []
    for batch in loader:
        case_ids.extend(batch.pop("case_id"))
        labels = batch.pop("labels")
        inputs = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(**inputs).logits
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        gold_ids.extend(labels.numpy().tolist())
        predicted_ids.extend(probs.argmax(axis=1).tolist())
        probabilities.extend(probs.tolist())
    return {
        "accuracy": float(accuracy_score(gold_ids, predicted_ids)),
        "macro_f1": float(f1_score(gold_ids, predicted_ids, labels=list(range(len(LABELS))), average="macro")),
        "mcc": float(matthews_corrcoef(gold_ids, predicted_ids)),
        "label_distribution": dict(Counter(LABELS[index] for index in gold_ids)),
        "prediction_distribution": dict(Counter(LABELS[index] for index in predicted_ids)),
        "case_ids": case_ids,
        "predicted_ids": predicted_ids,
        "probabilities": probabilities,
    }


def valid_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("target", {}).get("label", "")).lower() in LABEL_TO_ID]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
