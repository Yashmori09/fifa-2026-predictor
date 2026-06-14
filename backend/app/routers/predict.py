from fastapi import APIRouter, HTTPException
from app.schemas import PredictRequest, PredictResponse
from app.core.predictor import (
    predict_match_cached, get_lambdas_cached, most_likely_scoreline,
)

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("/", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.home_team == req.away_team:
        raise HTTPException(status_code=400, detail="Teams must be different")

    try:
        # Hybrid model: predict (λ_h, λ_a) directly, derive W/D/L + scoreline from
        # the Dixon-Coles-corrected joint Poisson matrix.
        ph, pd, pa = predict_match_cached(req.home_team, req.away_team, neutral=True)
        lam_h, lam_a = get_lambdas_cached(req.home_team, req.away_team)
        home_goals, away_goals = most_likely_scoreline(lam_h, lam_a)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if ph >= pd and ph >= pa:
        outcome = "home_win"
    elif pa >= ph and pa >= pd:
        outcome = "away_win"
    else:
        outcome = "draw"

    return PredictResponse(
        home_team=req.home_team,
        away_team=req.away_team,
        home_win=round(ph, 4),
        draw=round(pd, 4),
        away_win=round(pa, 4),
        predicted_outcome=outcome,
        home_goals=home_goals,
        away_goals=away_goals,
    )
