from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.inference import TicketClassifier

app = FastAPI(
    title="Customer Ticket Triage API",
    description="NLP classification for automated ticket routing and urgency detection.",
    version="1.0.0",
)

classifier = None


@app.on_event("startup")
def startup_event():
    global classifier
    try:
        classifier = TicketClassifier()
    except Exception as exc:
        print(f"Service running without pre-loaded weights: {exc}")
        classifier = None


class TicketRequest(BaseModel):
    ticket_id: str = Field(..., example="TCK-94812")
    message: str = Field(
        ...,
        min_length=5,
        example="Our production application is completely down with 500 errors after updating the credentials.",
    )


class TriageResponse(BaseModel):
    ticket_id: str
    predicted_intent: str
    confidence_score: float
    assigned_department: str
    urgency: str
    requires_escalation: bool


@app.get("/health")
def health():
    return {"status": "healthy", "model_ready": classifier is not None}


@app.post("/triage", response_model=TriageResponse)
def triage_ticket(payload: TicketRequest):
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Classification model artifact not loaded. Train the model first.",
        )

    result = classifier.predict(payload.message)
    return TriageResponse(
        ticket_id=payload.ticket_id,
        predicted_intent=result["predicted_intent"],
        confidence_score=result["confidence_score"],
        assigned_department=result["assigned_department"],
        urgency=result["urgency"],
        requires_escalation=result["requires_escalation"],
    )