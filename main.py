import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from openai import OpenAI

# Initialize FastAPI
app = FastAPI()

# Configure OpenAI client to use aipipe.org
# Assuming aipipe uses the standard OpenAI-compatible /v1 endpoint
client = OpenAI(
    api_key=os.environ.get("AIPIPE_API_KEY"),
    base_url="https://api.aipipe.org/v1" 
)

# 1. Define Request Schema
class CommentRequest(BaseModel):
    comment: str

# 2. Define Strict Response Schema for OpenAI Structured Outputs
class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int = Field(ge=1, le=5, description="Sentiment intensity (5=highly positive, 1=highly negative)")

@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):
    if not request.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")
        
    try:
        # Use the beta.chat.completions.parse method for guaranteed structured output
        response = client.beta.chat.completions.parse(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a precise sentiment analysis engine. Analyze the following comment and extract the overall sentiment and a 1-5 rating."},
                {"role": "user", "content": request.comment}
            ],
            response_format=SentimentResponse,
        )
        
        # Extract the parsed Pydantic object
        result = response.choices[0].message.parsed
        return result

    except Exception as e:
        # Handle API failures or parsing errors gracefully
        raise HTTPException(status_code=500, detail=f"AI Processing Error: {str(e)}")
