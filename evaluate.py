from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.dataset import INTENTS
from src.text_preprocessing import clean_ticket_text


def run_evaluation():
    test_file = Path("artifacts/test_tickets.csv")
    model_dir = Path("artifacts/triage_model")

    if not test_file.exists() or not model_dir.exists():
        raise FileNotFoundError("Artifacts missing. Run `python -m src.train` first.")

    df = pd.read_csv(test_file)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    clean_texts = [clean_ticket_text(t) for t in df["text"].tolist()]
    encodings = tokenizer(
        clean_texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**encodings)
        predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

    label_to_id = {v: k for k, v in model.config.id2label.items()}
    true_labels = [label_to_id[intent] for intent in df["intent"].tolist()]

    print("================ Model Evaluation Metrics ================")
    print(
        classification_report(
            true_labels,
            predictions,
            target_names=[model.config.id2label[i] for i in range(len(INTENTS))],
            digits=4,
        )
    )
    print("Confusion Matrix:")
    print(confusion_matrix(true_labels, predictions))
    print("==========================================================")


if __name__ == "__main__":
    run_evaluation()