from src.dataset import generate_ticket_corpus
from src.text_preprocessing import clean_ticket_text


def test_clean_ticket_text():
    raw_sample = "Check invoice #54210 and pay $120 at https://pay.example.com immediately."
    cleaned = clean_ticket_text(raw_sample)
    assert "[ID]" in cleaned
    assert "[AMOUNT]" in cleaned
    assert "[URL]" in cleaned
    assert "#54210" not in cleaned
    assert "$120" not in cleaned


def test_synthetic_data_generation():
    df = generate_ticket_corpus(n_samples=50, seed=1)
    assert len(df) == 50
    assert set(df.columns) == {"ticket_id", "text", "intent", "urgency", "target_department"}
    assert df["intent"].nunique() > 1