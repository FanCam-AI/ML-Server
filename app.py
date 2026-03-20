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
def process_result_task(data):
    return process_result(data)

@app.get("/ping")
async def health_check():
    return {"status": "healthy"}

@app.post("/process_run")
async def process_run(request: Request):
    data = await request.json()
    try:
        process_result_task.remote(data)  # 작업을 백그라운드에서 실행
    except Exception as e:
        # Ray 실행 자체가 실패하면 에러 기록
        return {"status": "failed", "error": str(e)}
    return {"status": "ok"}



uvicorn.run(app, host="0.0.0.0", port=settings.PORT)