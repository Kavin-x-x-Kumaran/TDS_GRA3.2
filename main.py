from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
import os

app = FastAPI()

# Initialize OpenAI Client
# Ensure OPENAI_API_KEY is set in your environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. Define the Structured Output Schema
class SentimentResponse(BaseModel):
    sentiment: str = Field(description="One of: 'positive', 'negative', 'neutral'")
    rating: int = Field(description="Sentiment intensity from 1 to 5")

# 2. Define the Request Schema
class CommentRequest(BaseModel):
    comment: str

@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):
    if not request.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")

    try:
        # Use OpenAI Structured Outputs
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini", # Note: gpt-4o-mini is the standard naming for the mini model
            messages=[
                {"role": "system", "content": "Analyze the sentiment of the provided comment. Categorize as positive, negative, or neutral and provide a rating from 1-5."},
                {"role": "user", "content": request.comment},
            ],
            response_format=SentimentResponse,
        )

        # Extract the parsed structured data
        return completion.choices[0].message.parsed

    except Exception as e:
        # Handle API failures or parsing errors
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
