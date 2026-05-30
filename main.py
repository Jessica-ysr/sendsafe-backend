from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import json
import os

app = FastAPI()

message_count = 0
low_conflict_count = 0

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class MessageRequest(BaseModel):
    message: str

@app.post("/analyze")
def analyze_message(request: MessageRequest):
    global message_count, low_conflict_count
    message_count += 1

    prompt = f"""
    Analyze this message for conflict potential:
    "{request.message}"
    
    Respond in this exact JSON format:
    {{
        "conflict_score": <number 0-100>,
        "feedback": "<one sentence explaining the score>",
        "aggressive_phrases": ["<phrase1>", "<phrase2>"],
        "rewrite": "<a calmer version of the message>"
    }}
    
    Only respond with JSON. No extra text.
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    result = json.loads(response.choices[0].message.content)
    if result.get('conflict_score', 100) < 40:
        low_conflict_count += 1
    return result

@app.get("/count")
def get_count():
    prevented_pct = round((low_conflict_count / message_count) * 100) if message_count > 0 else 0
    return {"count": message_count, "prevented": prevented_pct}