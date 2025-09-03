import requests
from dotenv import load_dotenv
from os import getenv

load_dotenv()

def fetch_jobs(query, location="remote"):
    url = (
        f"https://api.adzuna.com/v1/api/jobs/in/search/1"
        f"?app_id={ADZUNA_APP_ID}&app_key={ADZUNA_KEY}"
        f"&results_per_page=10&what={query}&where={location}"
    )
    res = requests.get(url).json()
    jobs = []
    for j in res.get("results", []):
        jobs.append({
            "title": j["title"],
            "company": j["company"]["display_name"],
            "location": j["location"]["display_name"],
            "salary": j.get("salary_min"),
            "url": j["redirect_url"],
        })
    return jobs
