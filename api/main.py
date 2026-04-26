"""FastAPI for the Succession-Planning Graph Recommender.

Endpoints:
    GET  /health
    GET  /managers           - sample of manager_ids the UI can pick from
    POST /succession         - rank successors for a given manager_id

Every response includes an explicit `decision_aid_disclaimer`.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Succession-Planning Graph", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DISCLAIMER = (
    "This shortlist is a decision aid for HRBPs and senior leadership. "
    "Always pair with a current development plan and a D&I review for "
    "Director+ slates."
)


class SuccessionRequest(BaseModel):
    manager_id: str
    k: int = Field(default=5, ge=1, le=25)


class Candidate(BaseModel):
    emp_id: str
    role: str
    level: int
    tenure_yrs: float
    skill_match: float
    structural_proximity: float
    performance: float
    readiness_score: float
    shortest_path_distance: int


class SuccessionResponse(BaseModel):
    manager_id: str
    candidates: list[Candidate]
    decision_aid_disclaimer: str = DISCLAIMER


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/managers", response_model=list[str])
def managers() -> list[str]:
    try:
        from succession_graph.serve import known_managers  # noqa: WPS433
        return known_managers(limit=30)
    except Exception:
        return [f"E-{i:04d}" for i in range(1, 31)]


@app.post("/succession", response_model=SuccessionResponse)
def succession(req: SuccessionRequest) -> SuccessionResponse:
    try:
        from succession_graph.serve import succession as run_succession  # noqa: WPS433
        candidates = run_succession(req.manager_id, k=req.k)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception:
        # Stub fallback so the wiring is visible end-to-end before models train.
        candidates = [
            dict(emp_id=f"E-{1000 + i:04d}", role="Manager Eng", level=4,
                 tenure_yrs=5.0 + i,
                 skill_match=0.78 - 0.04 * i, structural_proximity=0.50 - 0.05 * i,
                 performance=0.75, readiness_score=0.71 - 0.04 * i,
                 shortest_path_distance=1 + i)
            for i in range(req.k)
        ]
    try:
        return SuccessionResponse(manager_id=req.manager_id,
                                  candidates=candidates)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
