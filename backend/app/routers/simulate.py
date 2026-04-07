from fastapi import APIRouter
from app.schemas import SimulateResponse
from app.core.simulator import simulate_tournament

router = APIRouter(prefix="/simulate", tags=["simulate"])


@router.post("/full", response_model=SimulateResponse)
def run_full_simulation():
    result = simulate_tournament()
    return result
