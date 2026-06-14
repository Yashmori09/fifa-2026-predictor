from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import predict, simulate

app = FastAPI(
    title="FIFA 2026 Predictor API",
    description="Hybrid Poisson goal-scoring model (XGB Poisson regressors + Dixon-Coles) with EA FC, ELO, and StatsBomb international tournament features",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://fifa-2026-predictor.vercel.app",
        "https://*.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(simulate.router)


@app.get("/health")
def health():
    return {"status": "ok", "model": "XGB Poisson + Dixon-Coles (Phase 3 hybrid)"}


@app.on_event("startup")
def startup():
    import threading
    from app.core.predictor import warm_cache

    def _warm():
        print("Warming probability cache for 48 WC teams...")
        warm_cache()
        print("Cache warmed — ready to serve.")

    threading.Thread(target=_warm, daemon=True).start()
