from fastapi import FastAPI, HTTPException, Request, Header, Depends
from pydantic import BaseModel
from video_processing_tasks import process_result
from config import settings
import ray
import uvicorn

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
ray.init(ignore_reinit_error=True)

def verify_api_key(authorization: str = Header(None)):
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

    if token != settings.RUNPOD_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    return token




@ray.remote(num_gpus=0.25)
def precision_process_result_task(data):
    return process_result(data)

@ray.remote(num_cpus=1)
def normal_process_result_task(data):
    return process_result(data)

@app.get("/ping")
async def health_check():
    return {"status": "healthy"}

@app.get("/cpu_ready")
async def ready(api_key: str = Depends(verify_api_key)):
    try:
        resources = ray.available_resources()

        if resources.get("CPU", 0) < 1:
            return {"status": "not_ready"}

        return {"status": "ready"}

    except Exception:
        return {"status": "not_ready"}


@app.get("/gpu_ready")
async def ready(api_key: str = Depends(verify_api_key)):
    try:
        resources = ray.available_resources()
        if resources.get("GPU", 0) < 0.25:
            return {"status": "not_ready"}

        return {"status": "ready"}

    except Exception:
        return {"status": "not_ready"}


@app.post("/process_run")
async def process_run(request: Request, api_key: str = Depends(verify_api_key)):
    try:
        data = await request.json()
        input_data = data.get("input", {})
        tracking_mode = input_data.get("tracking_mode")
        if tracking_mode == "normal":
            normal_process_result_task.remote(input_data)
        elif tracking_mode == "precision":
            precision_process_result_task.remote(input_data)
    except Exception:
        return {"status": "failed"}

    return {"status": "ok"}



uvicorn.run(app, host="0.0.0.0", port=settings.PORT)