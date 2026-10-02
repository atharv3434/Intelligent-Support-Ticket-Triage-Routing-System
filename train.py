import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.dataset import INTENTS, load_dataset
from src.text_preprocessing import BASE_MODEL, TicketDataset

LABEL_MAP = {intent: i for i, intent in enumerate(INTENTS)}
ID_TO_LABEL = {i: intent for intent, i in LABEL_MAP.items()}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": float(acc)}


def train():
    df = load_dataset()
    df["label"] = df["intent"].map(LABEL_MAP)

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_dataset = TicketDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer)
    val_dataset = TicketDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(INTENTS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_MAP,
    )

    artifacts_path = Path("artifacts/triage_model")
    artifacts_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir="./artifacts/checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir="./artifacts/logs",
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Beginning model fine-tuning...")
    trainer.train()

    print("Saving production artifacts...")
    model.save_pretrained(artifacts_path)
    tokenizer.save_pretrained(artifacts_path)

    with open(artifacts_path / "label_map.json", "w") as f:
        json.dump(LABEL_MAP, f)

    # Save a test split for standalone evaluation
    val_df.to_csv("artifacts/test_tickets.csv", index=False)
    print(f"Artifacts successfully stored at {artifacts_path}")


if __name__ == "__main__":
    train()