from pathlib import Path
import numpy as np
import pandas as pd

INTENTS = ["Billing", "Technical_Support", "Account_Access", "Cancellation"]
DEPARTMENTS = {
    "Billing": "Finance_Operations",
    "Technical_Support": "Tier_2_Engineering",
    "Account_Access": "IT_Identity",
    "Cancellation": "Retention_Team",
}

TEMPLATES = {
    "Billing": [
        "I was charged twice for my subscription this month on invoice #{inv}.",
        "My payment failed via credit card but funds were deducted from my bank.",
        "Can I get a VAT invoice receipt for my purchase on {date}?",
        "Why is there an unexpected renewal surcharge of ${amt} on my statement?",
    ],
    "Technical_Support": [
        "Getting an unexpected 500 internal server error when uploading large CSVs.",
        "The application crashes whenever I try to export reports to PDF format.",
        "WebSocket connection repeatedly drops after 30 seconds of inactivity.",
        "API endpoint /v1/sync returns 403 Forbidden even with a valid bearer token.",
    ],
    "Account_Access": [
        "I lost access to my 2FA device and cannot log into my workspace account.",
        "Password reset link emailed to me keeps saying token expired immediately.",
        "Need to transfer admin rights to a new team member's email address.",
        "My account is locked due to multiple failed login attempts from my new laptop.",
    ],
    "Cancellation": [
        "Please cancel my annual enterprise subscription immediately and confirm.",
        "We are downsizing our team and need to downgrade to the free tier.",
        "I want to terminate my service contract and request a prorated refund.",
        "How do I close my organization profile and delete all stored account data?",
    ],
}


def generate_ticket_corpus(n_samples: int = 2400, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    records = []

    # Distribution reflecting real support ticket volumes
    intent_weights = [0.35, 0.30, 0.20, 0.15]

    for i in range(n_samples):
        intent = np.random.choice(INTENTS, p=intent_weights)
        template = np.random.choice(TEMPLATES[intent])
        text = template.format(
            inv=np.random.randint(10000, 99999),
            amt=np.random.randint(15, 450),
            date="2026-03-15",
        )

        # Urgent flag: presence of deadline or high impact keywords
        urgency = "High" if any(w in text.lower() for w in ["crashes", "500", "locked", "immediately", "twice"]) else "Normal"

        records.append(
            {
                "ticket_id": f"TCK-{i+10000}",
                "text": text,
                "intent": intent,
                "urgency": urgency,
                "target_department": DEPARTMENTS[intent],
            }
        )

    return pd.DataFrame(records)


def load_dataset(file_path: str = "data/raw/support_tickets.csv") -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        df = generate_ticket_corpus()
        df.to_csv(path, index=False)
        return df
    return pd.read_csv(path)