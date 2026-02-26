from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
import os

app = FastAPI()

# 1. Initialize Client with AI Pipe details
# Replace 'YOUR_AIPIPE_TOKEN' with your actual token or set it in your env
AIPIPE_TOKEN ="eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjI0ZjIwMDM4MDZAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.Tc32nyG08aK34k6e2XgZF5Zl7b6Pk38FEhK9Icgmt0U"

client = OpenAI(
    base_url="https://api.aipipe.org/v1", 
    api_key=AIPIPE_TOKEN
)

# Define Structured Output Schema
class SentimentResponse(BaseModel):
    sentiment: str = Field(description="Must be 'positive', 'negative', or 'neutral'")
    rating: int = Field(description="Integer from 1 to 5")

class CommentRequest(BaseModel):
    comment: str

@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):
    if not request.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")

    try:
        # Requesting gpt-4o-mini via AI Pipe
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a sentiment analysis tool. Return JSON with 'sentiment' (positive, negative, neutral) and 'rating' (1-5)."},
                {"role": "user", "content": request.comment},
            ],
            response_format=SentimentResponse,
        )

        return completion.choices[0].message.parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API Pipe Error: {str(e)}")
