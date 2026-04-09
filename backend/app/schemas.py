from pydantic import BaseModel


# ── Predict endpoint ─────────────────────────────────────────

class PredictRequest(BaseModel):
    home_team: str
    away_team: str


class PredictResponse(BaseModel):
    home_team: str
    away_team: str
    home_win: float
    draw: float
    away_win: float
    predicted_outcome: str
    home_goals: int
    away_goals: int


# ── Simulate endpoint ────────────────────────────────────────

class GroupMatchResult(BaseModel):
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    home_pts: int
    away_pts: int
    prob_home: float
    prob_draw: float
    prob_away: float


class GroupTeamStanding(BaseModel):
    team: str
    pts: int
    gd: int
    gf: int
    ga: int
    mp: int
    w: int
    d: int
    l: int


class GroupResult(BaseModel):
    group: str
    matches: list[GroupMatchResult]
    table: list[GroupTeamStanding]


class KnockoutMatch(BaseModel):
    team1: str
    team2: str
    team1_goals: int
    team2_goals: int
    winner: str
    penalties: bool
    prob_team1: float
    prob_draw: float
    prob_team2: float


class SimulateResponse(BaseModel):
    champion: str
    groups: list[GroupResult]
    best_thirds: dict[str, str]
    knockout: dict[str, list[KnockoutMatch]]
