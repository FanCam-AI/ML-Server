from fastapi import FastAPI, HTTPException, Request, Header, Depends
from pydantic import BaseModel
from workers import PrecisionProcessor, NormalProcessor
from config import settings
import ray
import uvicorn
import asyncio
import torch
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
ray.init(ignore_reinit_error=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if settings.SERVERLESS_ENVIRONMENT:
    if device == "cpu":
        normal_workers = [NormalProcessor.remote() for _ in range(settings.NORMAL_WORKER_COUNT)]
    elif device == "cuda":
        precision_workers = [PrecisionProcessor.remote() for _ in range(settings.PRECISION_WORKER_COUNT)]

else:
    normal_workers = [NormalProcessor.remote() for _ in range(settings.NORMAL_WORKER_COUNT)]
    precision_workers = [PrecisionProcessor.remote() for _ in range(settings.PRECISION_WORKER_COUNT)]



def verify_api_key(authorization: str = Header(None)):
    if settings.SERVERLESS_ENVIRONMENT:
        return None

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header"
        )

    try:
        scheme, token = authorization.split()

        if scheme.lower() != "bearer":
            raise ValueError()

    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization format"
        )

    if token != settings.API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    return token


async def get_available_worker(workers):
    results = await asyncio.gather(
        *[w.is_available.remote() for w in workers]
    )
    available_count = sum(results)
    busy_count = len(results) - available_count

    for w, ok in zip(workers, results):
        if ok:
            return w, available_count, busy_count

    return None

@app.get("/ping")
async def health_check():
    return {"status": "healthy"}

@app.get("/cpu_ready")
async def ready(api_key: str = Depends(verify_api_key)):
    try:
        worker, available_count, busy_count = await get_available_worker(normal_workers)
        if worker is not None:
            return {"status": "ready", "available_count": available_count, "busy_count": busy_count}

        return {"status": "not_ready"}

    except Exception:
        return {"status": "not_ready"}


@app.get("/gpu_ready")
async def ready(api_key: str = Depends(verify_api_key)):
    try:
        worker, available_count, busy_count = await get_available_worker(precision_workers)
        if worker is not None:
            return {"status": "ready", "available_count": available_count, "busy_count": busy_count}

        return {"status": "not_ready"}

    except Exception:
        return {"status": "not_ready"}


@app.post("/process_run")
async def process_run(request: Request, api_key: str = Depends(verify_api_key)):
    try:
        data = await request.json()
        input_data = data.get("input", {})
        tracking_mode = input_data.get("tracking_mode")
        if tracking_mode == "normal":
            worker, available_count, busy_count = await get_available_worker(normal_workers)

            if worker is None:
                return {"status": "busy"}

            worker.process.remote(input_data)

        elif tracking_mode == "precision":

            worker, available_count, busy_count = await get_available_worker(precision_workers)

            if worker is None:
                return {"status": "busy"}

            worker.process.remote(input_data)

    except Exception:
        return {"status": "failed"}

    return {"status": "ok"}



uvicorn.run(app, host="0.0.0.0", port=settings.PORT)