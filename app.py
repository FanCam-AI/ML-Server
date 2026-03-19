# import os
# from fastapi import FastAPI, HTTPException, Request
# from pydantic import BaseModel
# from video_processing_tasks import process_result
# import ray
#
# app = FastAPI()
#
# @app.on_event("startup")
# def startup_event():
#     ray.init()
#
#
# @ray.remote
# def process_result_task(data):
#     return process_result(data)
#
# @app.get("/ping")
# async def health_check():
#     return {"status": "healthy"}
#
# @app.post("/process_run")
# async def process_run(request: Request):
#     data = await request.json()
#     process_result_task.remote(data)
#     return {"status": "ok"}
#
#
# if __name__ == "__main__":
#     import uvicorn
#
#     port = int(os.getenv("PORT", 80))
#
#     uvicorn.run(app, host="0.0.0.0", port=port)

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Create FastAPI app
app = FastAPI()

# Define request models
class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

class GenerationResponse(BaseModel):
    generated_text: str

# Global variable to track requests
request_count = 0

# Health check endpoint; required for Runpod to monitor worker health
@app.get("/ping")
async def health_check():
    return {"status": "healthy"}

# Our custom generation endpoint
@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    global request_count
    request_count += 1

    # A simple mock implementation; we'll replace this with an actual model later
    generated_text = f"Response to: {request.prompt} (request #{request_count})"

    return {"generated_text": generated_text}

# A simple endpoint to show request stats
@app.get("/stats")
async def stats():
    return {"total_requests": request_count}

# Run the app when the script is executed
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 80))
    print(f"Starting server on port {port}")

    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=port)