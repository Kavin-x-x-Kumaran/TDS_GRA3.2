import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from openai import OpenAI

app = FastAPI()

# Make sure there are no trailing slashes in the base_url unless required
client = OpenAI(
    api_key=os.environ.get("AIPIPE_API_KEY"),
    base_url="https://api.aipipe.org/v1" 
)

class CommentRequest(BaseModel):
    comment: str

class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int = Field(ge=1, le=5)

@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):
    if not request.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")
        
    try:
        # We'll use the standard completions here just in case the proxy 
        # doesn't fully support the .parse() beta method yet.
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Return JSON: {\"sentiment\": \"positive\"|\"negative\"|\"neutral\", \"rating\": 1-5}"},
                {"role": "user", "content": request.comment}
            ],
            response_format={ "type": "json_object" }
        )
        
        # Manually parse since some proxies strip metadata
        import json
        content = json.loads(response.choices[0].message.content)
        return content

    except Exception as e:
        # This will now tell us if it's a 401 (Auth), 404 (URL), or Timeout
        raise HTTPException(status_code=500, detail=f"Proxy Error: {type(e).__name__} - {str(e)}")
