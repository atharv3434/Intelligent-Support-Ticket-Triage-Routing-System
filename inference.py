from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.dataset import DEPARTMENTS
from src.text_preprocessing import clean_ticket_text


class TicketClassifier:

    def __init__(self, model_dir: str = "artifacts/triage_model"):
        self.model_path = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.eval()

    def assess_urgency(self, text: str) -> str:
        urgent_keywords = ["urgent", "asap", "down", "outage", "immediately", "loss", "critical", "500"]
        return "Critical" if any(w in text.lower() for w in urgent_keywords) else "Standard"

    def predict(self, ticket_text: str) -> dict:
        cleaned = clean_ticket_text(ticket_text)
        inputs = self.tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=128)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)

        pred_id = int(torch.argmax(probs))
        confidence = float(probs[pred_id])
        intent = self.model.config.id2label[pred_id]

        return {
            "predicted_intent": intent,
            "confidence_score": round(confidence, 4),
            "assigned_department": DEPARTMENTS.get(intent, "General_Support"),
            "urgency": self.assess_urgency(ticket_text),
            "requires_escalation": confidence < 0.65 or self.assess_urgency(ticket_text) == "Critical",
        }