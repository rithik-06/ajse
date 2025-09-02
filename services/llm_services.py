from openai import OpenAI
from utils.config import OPENAI_KEY

client = OpenAI(api_key=OPENAI_KEY)

def summarize_jobs(jobs, user_query):
    text = f"User Query: {user_query}\n\nJobs:\n"
    for j in jobs:
        text += f"- {j['title']} at {j['company']} in {j['location']} (Salary: {j['salary']})\nLink: {j['url']}\n\n"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful job search assistant."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content
