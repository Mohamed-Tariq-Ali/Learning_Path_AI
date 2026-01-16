# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field
# from typing import Optional
# from google import genai
# from google.genai import types
# import os
# from dotenv import load_dotenv
# load_dotenv()
# from enum import Enum
#
# # Initialize FastAPI app
# app = FastAPI(
#     title="Text Summarization API",
#     description="API for text summarization using Google Gemini AI",
#     version="1.0.0"
# )
#
# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# # Configure Gemini AI
# # Set your API key as an environment variable: GEMINI_API_KEY
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# client = None
# if GEMINI_API_KEY:
#     client = genai.Client(api_key=GEMINI_API_KEY)
#
#
# # Enums for request options
# class SummaryLength(str, Enum):
#     short = "short"
#     medium = "medium"
#     long = "long"
#
#
# class SummaryStyle(str, Enum):
#     bullet_points = "bullet_points"
#     paragraph = "paragraph"
#     key_points = "key_points"
#
#
# # Request models
# class SummarizeRequest(BaseModel):
#     text: str = Field(..., description="Text to summarize", min_length=50)
#     length: Optional[SummaryLength] = Field(
#         default=SummaryLength.medium,
#         description="Desired summary length"
#     )
#     style: Optional[SummaryStyle] = Field(
#         default=SummaryStyle.paragraph,
#         description="Summary style format"
#     )
#     language: Optional[str] = Field(
#         default="English",
#         description="Language for the summary"
#     )
#
#
# class SummarizeResponse(BaseModel):
#     summary: str
#     original_length: int
#     summary_length: int
#     compression_ratio: float
#
#
# # Helper function to build prompt
# def build_summarization_prompt(text: str, length: SummaryLength, style: SummaryStyle, language: str) -> str:
#     length_instructions = {
#         SummaryLength.short: "Keep it very concise, about 2-3 sentences.",
#         SummaryLength.medium: "Provide a moderate summary, about 1-2 paragraphs.",
#         SummaryLength.long: "Provide a detailed summary covering all main points."
#     }
#
#     style_instructions = {
#         SummaryStyle.bullet_points: "Format the summary as bullet points.",
#         SummaryStyle.paragraph: "Format the summary as flowing paragraphs.",
#         SummaryStyle.key_points: "Extract and list the key points with brief explanations."
#     }
#
#     prompt = f"""Summarize the following text in {language}.
#
# {length_instructions[length]}
# {style_instructions[style]}
#
# Text to summarize:
# {text}
#
# Summary:"""
#
#     return prompt
#
#
# @app.get("/")
# async def root():
#     """Root endpoint with API information"""
#     return {
#         "message": "Text Summarization API using Gemini AI",
#         "endpoints": {
#             "/summarize": "POST - Summarize text",
#             "/health": "GET - Check API health",
#             "/docs": "GET - API documentation"
#         }
#     }
#
#
# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     api_configured = client is not None
#     return {
#         "status": "healthy" if api_configured else "warning",
#         "gemini_api_configured": api_configured,
#         "message": "API key not configured" if not api_configured else "All systems operational"
#     }
#
#
# @app.post("/summarize", response_model=SummarizeResponse)
# async def summarize_text(request: SummarizeRequest):
#     """
#     Summarize text using Gemini AI
#
#     - **text**: The text to summarize (minimum 50 characters)
#     - **length**: short, medium, or long
#     - **style**: bullet_points, paragraph, or key_points
#     - **language**: Target language for summary (default: English)
#     """
#
#     if not client:
#         raise HTTPException(
#             status_code=500,
#             detail="Gemini API key not configured. Set GEMINI_API_KEY environment variable."
#         )
#
#     try:
#         # Build prompt
#         prompt = build_summarization_prompt(
#             request.text,
#             request.length,
#             request.style,
#             request.language
#         )
#
#         # Generate summary using new API
#         response = client.models.generate_content(
#             model='gemini-2.5-flash',
#             contents=prompt
#         )
#         summary = response.text
#
#         # Calculate metrics
#         original_length = len(request.text)
#         summary_length = len(summary)
#         compression_ratio = round(summary_length / original_length, 2)
#
#         return SummarizeResponse(
#             summary=summary,
#             original_length=original_length,
#             summary_length=summary_length,
#             compression_ratio=compression_ratio
#         )
#
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error generating summary: {str(e)}"
#         )
#
#
# @app.post("/summarize/quick")
# async def quick_summarize(text: str):
#     """
#     Quick summarization with default settings
#
#     - **text**: The text to summarize
#     """
#
#     if len(text) < 50:
#         raise HTTPException(
#             status_code=400,
#             detail="Text must be at least 50 characters long"
#         )
#
#     request = SummarizeRequest(text=text)
#     return await summarize_text(request)
#
#
# if __name__ == "__main__":
#     import uvicorn
#
#     uvicorn.run(app, host="0.0.0.0", port=8000)

#========================== TEXT SUMMARIZATION FROM FILES USING AI======================================

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from google import genai
from google.genai import types
import os
from enum import Enum
import io
import pandas as pd
from pypdf import PdfReader

# Initialize FastAPI app
app = FastAPI(
    title="Text Summarization API",
    description="API for text summarization using Google Gemini AI",
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

# Configure Gemini AI
# Set your API key as an environment variable: GEMINI_API_KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)


# Enums for request options
class SummaryLength(str, Enum):
    short = "short"
    medium = "medium"
    long = "long"


class SummaryStyle(str, Enum):
    bullet_points = "bullet_points"
    paragraph = "paragraph"
    key_points = "key_points"


# Request models
class SummarizeRequest(BaseModel):
    text: str = Field(..., description="Text to summarize", min_length=50)
    length: Optional[SummaryLength] = Field(
        default=SummaryLength.medium,
        description="Desired summary length"
    )
    style: Optional[SummaryStyle] = Field(
        default=SummaryStyle.paragraph,
        description="Summary style format"
    )
    language: Optional[str] = Field(
        default="English",
        description="Language for the summary"
    )


class SummarizeResponse(BaseModel):
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float


class FileSummaryResponse(BaseModel):
    filename: str
    file_type: str
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float
    error: Optional[str] = None


# Helper function to build prompt
def build_summarization_prompt(text: str, length: SummaryLength, style: SummaryStyle, language: str) -> str:
    length_instructions = {
        SummaryLength.short: "Keep it very concise, about 2-3 sentences.",
        SummaryLength.medium: "Provide a moderate summary, about 1-2 paragraphs.",
        SummaryLength.long: "Provide a detailed summary covering all main points."
    }

    style_instructions = {
        SummaryStyle.bullet_points: "Format the summary as bullet points.",
        SummaryStyle.paragraph: "Format the summary as flowing paragraphs.",
        SummaryStyle.key_points: "Extract and list the key points with brief explanations."
    }

    prompt = f"""Summarize the following text in {language}.

{length_instructions[length]}
{style_instructions[style]}

Text to summarize:
{text}

Summary:"""

    return prompt


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Text Summarization API using Gemini AI",
        "endpoints": {
            "/summarize": "POST - Summarize text",
            "/health": "GET - Check API health",
            "/docs": "GET - API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    api_configured = client is not None
    return {
        "status": "healthy" if api_configured else "warning",
        "gemini_api_configured": api_configured,
        "message": "API key not configured" if not api_configured else "All systems operational"
    }


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(request: SummarizeRequest):
    """
    Summarize text using Gemini AI

    - **text**: The text to summarize (minimum 50 characters)
    - **length**: short, medium, or long
    - **style**: bullet_points, paragraph, or key_points
    - **language**: Target language for summary (default: English)
    """

    if not client:
        raise HTTPException(
            status_code=500,
            detail="Gemini API key not configured. Set GEMINI_API_KEY environment variable."
        )

    try:
        # Build prompt
        prompt = build_summarization_prompt(
            request.text,
            request.length,
            request.style,
            request.language
        )

        # Generate summary using new API
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        summary = response.text

        # Calculate metrics
        original_length = len(request.text)
        summary_length = len(summary)
        compression_ratio = round(summary_length / original_length, 2)

        return SummarizeResponse(
            summary=summary,
            original_length=original_length,
            summary_length=summary_length,
            compression_ratio=compression_ratio
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating summary: {str(e)}"
        )


@app.post("/summarize/quick")
async def quick_summarize(text: str):
    """
    Quick summarization with default settings

    - **text**: The text to summarize
    """

    if len(text) < 50:
        raise HTTPException(
            status_code=400,
            detail="Text must be at least 50 characters long"
        )

    request = SummarizeRequest(text=text)
    return await summarize_text(request)


@app.post("/summarize/files")
async def summarize_files(
        files: List[UploadFile] = File(...),
        length: SummaryLength = SummaryLength.medium,
        style: SummaryStyle = SummaryStyle.paragraph,
        language: str = "English"
):
    """
    Upload and summarize multiple files (PDF, Excel)

    - **files**: Multiple files to upload and summarize
    - **length**: Desired summary length (short, medium, long)
    - **style**: Summary style (bullet_points, paragraph, key_points)
    - **language**: Target language for summary
    """

    if not client:
        raise HTTPException(
            status_code=500,
            detail="Gemini API key not configured. Set GEMINI_API_KEY environment variable."
        )

    results = []

    for file in files:
        try:
            content = await file.read()
            name = file.filename.lower()
            extracted_text = ""
            file_type = "UNKNOWN"

            # Extract text based on file type
            if name.endswith((".xls", ".xlsx")):
                df = pd.read_excel(io.BytesIO(content))
                # Convert DataFrame to text representation
                extracted_text = df.to_string()
                file_type = "EXCEL"

            elif name.endswith(".pdf"):
                reader = PdfReader(io.BytesIO(content))
                # Extract text from all pages with better error handling
                text_parts = []
                for page in reader.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception as e:
                        print(f"Error extracting page: {e}")
                        continue
                extracted_text = "\n".join(text_parts)
                file_type = "PDF"

            else:
                results.append(FileSummaryResponse(
                    filename=file.filename,
                    file_type="ERROR",
                    summary="",
                    original_length=0,
                    summary_length=0,
                    compression_ratio=0.0,
                    error="Unsupported file type. Only PDF and Excel files are supported."
                ))
                continue

            # Check if text was extracted
            if not extracted_text or len(extracted_text.strip()) < 50:
                results.append(FileSummaryResponse(
                    filename=file.filename,
                    file_type=file_type,
                    summary="",
                    original_length=len(extracted_text),
                    summary_length=0,
                    compression_ratio=0.0,
                    error="Could not extract sufficient text from file."
                ))
                continue

            # Generate summary
            prompt = build_summarization_prompt(
                extracted_text,
                length,
                style,
                language
            )

            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            summary = response.text

            # Calculate metrics
            original_length = len(extracted_text)
            summary_length = len(summary)
            compression_ratio = round(summary_length / original_length, 2)

            results.append(FileSummaryResponse(
                filename=file.filename,
                file_type=file_type,
                summary=summary,
                original_length=original_length,
                summary_length=summary_length,
                compression_ratio=compression_ratio
            ))

        except Exception as e:
            results.append(FileSummaryResponse(
                filename=file.filename,
                file_type="ERROR",
                summary="",
                original_length=0,
                summary_length=0,
                compression_ratio=0.0,
                error=f"Error processing file: {str(e)}"
            ))

    return {
        "total_files": len(results),
        "successful": len([r for r in results if not r.error]),
        "failed": len([r for r in results if r.error]),
        "results": results
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)