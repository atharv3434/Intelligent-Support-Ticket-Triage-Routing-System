import re
import torch
from transformers import AutoTokenizer

BASE_MODEL = "distilbert-base-uncased"


def clean_ticket_text(text: str) -> str:
    """Removes noise while preserving semantic punctuation and tokens."""
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
    text = re.sub(r"#\d+", "[ID]", text)
    text = re.sub(r"\$\d+", "[AMOUNT]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class TicketDataset(torch.utils.data.Dataset):

    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = [clean_ticket_text(t) for t in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item