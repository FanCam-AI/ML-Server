import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from video_processing_tasks import process_result
from config import settings
import ray
import uvicorn
app = FastAPI()

@app.on_event("startup")
def startup_event():
    ray.init()


@ray.remote
def process_result_task(data):
    return process_result(data)

@app.get("/ping")
async def health_check():
    return {"status": "healthy"}

@app.post("/process_run")
async def process_run(request: Request):
    data = await request.json()
    process_result_task.remote(data)
    return {"status": "ok"}



uvicorn.run(app, host="0.0.0.0", port=settings.PORT)