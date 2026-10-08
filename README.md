# 🎫 Customer Ticket Triage API

An automated NLP-powered customer support triage service built with **FastAPI**. It classifies incoming customer support tickets by intent, assigns them to appropriate departments, determines ticket urgency, and flags critical issues requiring immediate escalation.

---

## 📌 Features

- **Intent Classification**: Identifies user problem types (e.g., billing, outage, technical bug, account access).
- **Automated Routing**: Maps predicted intent directly to the responsible team or department.
- **Urgency & Escalation Flagging**: Assesses sentiment and severity to flag high-priority/escalation-level tickets.
- **Confidence Scoring**: Returns a model certainty score for threshold-based human-in-the-loop workflows.
- **RESTful API with OpenAPI/Swagger**: Fully typed request/response validation using Pydantic models.
- **Health Monitoring**: Dedicated health check endpoint tracking model artifact status.

---
