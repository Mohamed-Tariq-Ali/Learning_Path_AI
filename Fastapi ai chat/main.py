from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Mini LLM Gateway - Gemini",
    description="A simple gateway to Gemini API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


# Data Models
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 1000
    temperature: float = 0.7


class ChatResponse(BaseModel):
    response: str
    tokens_used: int
    timestamp: str


# Endpoints
@app.get("/")
async def health_check():
    return {
        "status": "running",
        "service": "Mini LLM Gateway - Gemini",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    logger.info(f"Received chat request with {len(request.messages)} messages")

    try:
        # Convert messages to Gemini format
        gemini_contents = []
        for msg in request.messages:
            role = "model" if msg.role == "assistant" else "user"
            gemini_contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=msg.content)]
                )
            )

        # Configure generation settings
        config = types.GenerateContentConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
        )

        # Generate response
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=gemini_contents,
            config=config
        )

        response_text = response.text

        # Get token usage
        tokens_used = response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0

        logger.info(f"Successfully generated response. Tokens used: {tokens_used}")

        return ChatResponse(
            response=response_text,
            tokens_used=tokens_used,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}", exc_info=True)
        # Return more detailed error information
        error_detail = {
            "error": str(e),
            "type": type(e).__name__,
            "message": "Check server logs for details"
        }
        raise HTTPException(status_code=500, detail=error_detail)


# if __name__ == "__main__":
#     import uvicorn
#
#     uvicorn.run(app, host="0.0.0.0", port=8000)