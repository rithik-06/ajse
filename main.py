from fastapi import FastAPI
from services.jobs_service import fetch_jobs
from services.llm_service import summarize_jobs

app = FastAPI()

@app.get("/")
def home():
    return {"message": "AI Job Search Agent is running 🚀"}

@app.get("/search_jobs")
def search_jobs(role: str, location: str = "remote"):
    jobs = fetch_jobs(role, location)
    summary = summarize_jobs(jobs, role)
    return {"query": role, "results": jobs, "summary": summary}
