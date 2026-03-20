import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from video_processing_tasks import process_result
from config import settings
import ray
import uvicorn
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    import ray
    ray.init(ignore_reinit_error=True)

@ray.remote(num_gpus=0.25)
def precision_process_result_task(data):
    return process_result(data)

@ray.remote(num_cpus=1)
def normal_process_result_task(data):
    return process_result(data)

@app.get("/ping")
async def health_check():
    return {"status": "healthy"}

@app.post("/process_run")
async def process_run(request: Request):
    try:
        data = await request.json()
        input_data = data.get("input", {})
        tracking_mode = input_data.get("tracking_mode")
        if tracking_mode == "normal":
            normal_process_result_task(input_data)
        elif tracking_mode == "precision":
            precision_process_result_task.remote(input_data)
    except Exception as e:
        return {"status": "failed", "error": str(e)}
    return {"status": "ok"}



uvicorn.run(app, host="0.0.0.0", port=settings.PORT)