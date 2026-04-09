from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import predict, simulate

app = FastAPI(
    title="FIFA 2026 Predictor API",
    description="XGB×3 + RF×1 ensemble match prediction and tournament simulation (EA + ELO, no DC)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(simulate.router)


@app.get("/health")
def health():
    return {"status": "ok", "model": "XGB*3 + RF*1"}


@app.on_event("startup")
def startup():
    import threading
    from app.core.predictor import warm_cache

    def _warm():
        print("Warming probability cache for 48 WC teams...")
        warm_cache()
        print("Cache warmed — ready to serve.")

    threading.Thread(target=_warm, daemon=True).start()
