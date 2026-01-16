# # """
# # Production-ready FastAPI server for Python Coding Assistant
# # Install: pip install fastapi uvicorn transformers peft torch pydantic
# #
# # Run: uvicorn server:app --host 0.0.0.0 --port 8000 --reload
# # """
# #
# # from fastapi import FastAPI, HTTPException
# # from fastapi.middleware.cors import CORSMiddleware
# # from pydantic import BaseModel, Field
# # from transformers import AutoTokenizer, AutoModelForCausalLM
# # from peft import PeftModel
# # import torch
# # from typing import Optional, List
# # import logging
# # from datetime import datetime
# #
# # # Setup logging
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)
# #
# # # ============= CONFIGURATION =============
# # MODEL_PATH = "./python-coding-assistant"
# # BASE_MODEL = "microsoft/phi-2"
# #
# # # ============= INITIALIZE APP =============
# # app = FastAPI(
# #     title="Python Coding Assistant API",
# #     description="AI-powered Python coding helper",
# #     version="1.0.0"
# # )
# #
# # # CORS middleware
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],  # In production, specify your domains
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )
# #
# #
# # # ============= MODELS FOR API =============
# # class ChatRequest(BaseModel):
# #     question: str = Field(..., description="Your Python coding question")
# #     max_tokens: int = Field(200, ge=50, le=1000, description="Max tokens to generate")
# #     temperature: float = Field(0.7, ge=0.1, le=2.0, description="Sampling temperature")
# #
# #     class Config:
# #         schema_extra = {
# #             "example": {
# #                 "question": "How do I create a virtual environment in Python?",
# #                 "max_tokens": 200,
# #                 "temperature": 0.7
# #             }
# #         }
# #
# #
# # class ChatResponse(BaseModel):
# #     answer: str
# #     tokens_used: int
# #     model: str
# #     timestamp: str
# #
# #
# # class HealthResponse(BaseModel):
# #     status: str
# #     model_loaded: bool
# #     device: str
# #     uptime: str
# #
# #
# # # ============= GLOBAL STATE =============
# # class ModelState:
# #     def __init__(self):
# #         self.model = None
# #         self.tokenizer = None
# #         self.device = None
# #         self.start_time = datetime.now()
# #
# #
# # state = ModelState()
# #
# #
# # # ============= LOAD MODEL ON STARTUP =============
# # @app.on_event("startup")
# # async def load_model():
# #     """Load model when server starts"""
# #     try:
# #         logger.info("Loading model and tokenizer...")
# #
# #         # Determine device
# #         state.device = "cuda" if torch.cuda.is_available() else "cpu"
# #         logger.info(f"Using device: {state.device}")
# #
# #         # Load tokenizer
# #         state.tokenizer = AutoTokenizer.from_pretrained(
# #             BASE_MODEL,
# #             trust_remote_code=True
# #         )
# #         state.tokenizer.pad_token = state.tokenizer.eos_token
# #
# #         # Load base model
# #         base_model = AutoModelForCausalLM.from_pretrained(
# #             BASE_MODEL,
# #             trust_remote_code=True,
# #             torch_dtype=torch.float16 if state.device == "cuda" else torch.float32,
# #             device_map="auto"
# #         )
# #
# #         # Load LoRA weights
# #         state.model = PeftModel.from_pretrained(base_model, MODEL_PATH)
# #         state.model.eval()  # Set to evaluation mode
# #
# #         logger.info("✅ Model loaded successfully!")
# #
# #     except Exception as e:
# #         logger.error(f"❌ Failed to load model: {e}")
# #         raise
# #
# #
# # # ============= HELPER FUNCTIONS =============
# # def format_prompt(question: str) -> str:
# #     """Format the question in the training format"""
# #     return f"### Question: {question}\n### Answer:"
# #
# #
# # def generate_answer(question: str, max_tokens: int = 200, temperature: float = 0.7) -> tuple:
# #     """Generate answer using the model"""
# #     if state.model is None:
# #         raise ValueError("Model not loaded")
# #
# #     # Format and tokenize
# #     prompt = format_prompt(question)
# #     inputs = state.tokenizer(
# #         prompt,
# #         return_tensors="pt",
# #         truncation=True,
# #         max_length=512
# #     ).to(state.device)
# #
# #     # Generate
# #     with torch.no_grad():
# #         outputs = state.model.generate(
# #             **inputs,
# #             max_new_tokens=max_tokens,
# #             temperature=temperature,
# #             do_sample=True,
# #             top_p=0.9,
# #             top_k=50,
# #             repetition_penalty=1.1,
# #             pad_token_id=state.tokenizer.eos_token_id,
# #             eos_token_id=state.tokenizer.eos_token_id,
# #         )
# #
# #     # Decode
# #     full_response = state.tokenizer.decode(outputs[0], skip_special_tokens=True)
# #
# #     # Extract just the answer part
# #     if "### Answer:" in full_response:
# #         answer = full_response.split("### Answer:")[-1].strip()
# #     else:
# #         answer = full_response.strip()
# #
# #     tokens_used = outputs[0].shape[0]
# #
# #     return answer, tokens_used
# #
# #
# # # ============= API ENDPOINTS =============
# # @app.get("/", response_model=dict)
# # async def root():
# #     """Welcome endpoint"""
# #     return {
# #         "message": "Welcome to Python Coding Assistant API",
# #         "version": "1.0.0",
# #         "endpoints": {
# #             "/chat": "POST - Ask a Python coding question",
# #             "/health": "GET - Check API health",
# #             "/docs": "GET - Interactive API documentation"
# #         }
# #     }
# #
# #
# # @app.get("/health", response_model=HealthResponse)
# # async def health_check():
# #     """Health check endpoint"""
# #     uptime = datetime.now() - state.start_time
# #
# #     return HealthResponse(
# #         status="healthy" if state.model is not None else "unhealthy",
# #         model_loaded=state.model is not None,
# #         device=state.device if state.device else "unknown",
# #         uptime=str(uptime).split('.')[0]  # Remove microseconds
# #     )
# #
# #
# # @app.post("/chat", response_model=ChatResponse)
# # async def chat(request: ChatRequest):
# #     """
# #     Ask the AI a Python coding question
# #
# #     Example:
# #     ```json
# #     {
# #         "question": "How do I use async/await in Python?",
# #         "max_tokens": 200,
# #         "temperature": 0.7
# #     }
# #     ```
# #     """
# #     try:
# #         if state.model is None:
# #             raise HTTPException(status_code=503, detail="Model not loaded")
# #
# #         logger.info(f"Processing question: {request.question[:50]}...")
# #
# #         # Generate answer
# #         answer, tokens = generate_answer(
# #             request.question,
# #             request.max_tokens,
# #             request.temperature
# #         )
# #
# #         return ChatResponse(
# #             answer=answer,
# #             tokens_used=tokens,
# #             model=MODEL_PATH,
# #             timestamp=datetime.now().isoformat()
# #         )
# #
# #     except Exception as e:
# #         logger.error(f"Error processing request: {e}")
# #         raise HTTPException(status_code=500, detail=str(e))
# #
# #
# # @app.post("/batch")
# # async def batch_questions(questions: List[str], max_tokens: int = 200):
# #     """Process multiple questions at once"""
# #     try:
# #         results = []
# #         for q in questions:
# #             answer, tokens = generate_answer(q, max_tokens)
# #             results.append({
# #                 "question": q,
# #                 "answer": answer,
# #                 "tokens": tokens
# #             })
# #         return {"results": results}
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=str(e))
# #
# #
# # @app.get("/examples")
# # async def get_examples():
# #     """Get example questions you can ask"""
# #     return {
# #         "examples": [
# #             "How do I create a virtual environment in Python?",
# #             "What's the difference between append and extend?",
# #             "How do I use async/await in FastAPI?",
# #             "How do I read a JSON file in Python?",
# #             "What are Python decorators and how do I use them?",
# #             "How do I handle exceptions in Python?",
# #             "How do I use list comprehensions?",
# #             "How do I connect to a database in Python?",
# #             "What's the best way to make HTTP requests?",
# #             "How do I use type hints in Python?"
# #         ]
# #     }
# #
# #
# # # ============= ERROR HANDLERS =============
# # @app.exception_handler(Exception)
# # async def global_exception_handler(request, exc):
# #     logger.error(f"Global error: {exc}")
# #     return {
# #         "error": "Internal server error",
# #         "detail": str(exc)
# #     }
# #
# #
# # # ============= RUN SERVER =============
# # if __name__ == "__main__":
# #     import uvicorn
# #
# #     uvicorn.run(
# #         app,
# #         host="0.0.0.0",
# #         port=8000,
# #         log_level="info"
# #     )



import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from fastapi import FastAPI
import logging
from pathlib import Path

from fastapi.middleware.cors import CORSMiddleware


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow any frontend origin
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== PATHS =====
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PATH = PROJECT_ROOT / "models" / "python-coding-assistant"
BASE_MODEL = "microsoft/phi-2"

device = "cpu"  # change to "cuda" if GPU available

@app.on_event("startup")
def load_model():
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
        device_map="cpu"
    )

    logger.info("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH
    )

    model.eval()

    app.state.tokenizer = tokenizer
    app.state.model = model

    logger.info("✅ Model loaded successfully")

@app.post("/chat")
def generate(prompt: str):
    tokenizer = app.state.tokenizer
    model = app.state.model

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ========== CLEAN RESPONSE ==========
    # Remove unwanted newlines and stray quotes
    answer = decoded.replace('\n"', '\n').replace('\r', '').strip()

    # Optional: remove multiple consecutive newlines
    import re
    answer = re.sub(r'\n+', '\n', answer)

    return {
        "response": answer
    }
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, Field
# from pathlib import Path
# from datetime import datetime
# import logging
#
# # ================= LOGGING =================
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("server")
#
# # ================= PATHS =================
# PROJECT_ROOT = Path(__file__).resolve().parents[0]  # adjust if needed
# ADAPTER_PATH = PROJECT_ROOT / "models" / "python-coding-assistant"
# BASE_MODEL = "microsoft/phi-2"
# MAX_INPUT_LENGTH = 512
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#
# # ================= FASTAPI =================
# app = FastAPI(
#     title="Python Coding Assistant API",
#     version="1.0.0",
#     description="Production-ready Python coding assistant"
# )
#
# # ================= STATE =================
# class ModelState:
#     model = None
#     tokenizer = None
#     start_time = datetime.now()
#
# state = ModelState()
#
# # ================= SCHEMAS =================
# class ChatRequest(BaseModel):
#     question: str = Field(..., min_length=3)
#     max_tokens: int = Field(256, ge=50, le=1024)
#     temperature: float = Field(0.3, ge=0.1, le=1.5)
#
# class ChatResponse(BaseModel):
#     answer: str
#     tokens_used: int
#     model: str
#     timestamp: str
#
# class HealthResponse(BaseModel):
#     status: str
#     model_loaded: bool
#     device: str
#     uptime: str
#
# # ================= STARTUP =================
# @app.on_event("startup")
# def load_model():
#     try:
#         logger.info("🔄 Loading tokenizer...")
#         state.tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
#         state.tokenizer.pad_token = state.tokenizer.eos_token
#
#         logger.info("🔄 Loading base model...")
#         base_model = AutoModelForCausalLM.from_pretrained(
#             BASE_MODEL,
#             torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32,
#             device_map="auto"
#         )
#
#         logger.info("🔄 Loading LoRA adapter...")
#         state.model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
#         state.model.eval()
#
#         logger.info(f"✅ Model loaded successfully on {DEVICE}")
#
#     except Exception as e:
#         logger.exception("❌ Model loading failed")
#         raise RuntimeError(str(e))
#
# # ================= HELPERS =================
# def format_prompt(question: str) -> str:
#     return f"""### Instruction:
# You are a helpful Python programming assistant.
# Answer clearly and concisely.
#
# ### Question:
# {question}
#
# ### Answer:
# """
#
# def generate_answer(question: str, max_tokens: int, temperature: float):
#     prompt = format_prompt(question)
#
#     inputs = state.tokenizer(
#         prompt,
#         return_tensors="pt",
#         truncation=True,
#         max_length=MAX_INPUT_LENGTH,
#         padding=False
#     ).to(DEVICE)
#
#     with torch.inference_mode():
#         output = state.model.generate(
#             **inputs,
#             max_new_tokens=max_tokens,
#             temperature=temperature,
#             do_sample=True,
#             top_p=0.9,
#             top_k=50,
#             repetition_penalty=1.1,
#             pad_token_id=state.tokenizer.eos_token_id,
#             eos_token_id=state.tokenizer.eos_token_id,
#         )
#
#     decoded = state.tokenizer.decode(output[0], skip_special_tokens=True)
#     if "### Answer:" in decoded:
#         answer = decoded.split("### Answer:")[-1].strip()
#     else:
#         answer = decoded.strip()
#
#     tokens_used = output.shape[-1]
#     return answer, tokens_used
#
# # ================= ROUTES =================
# @app.get("/health", response_model=HealthResponse)
# def health():
#     uptime = datetime.now() - state.start_time
#     return HealthResponse(
#         status="healthy" if state.model else "unhealthy",
#         model_loaded=state.model is not None,
#         device=DEVICE,
#         uptime=str(uptime).split(".")[0]
#     )
#
# @app.post("/chat", response_model=ChatResponse)
# def chat(req: ChatRequest):
#     if state.model is None:
#         raise HTTPException(status_code=503, detail="Model not loaded")
#
#     answer, tokens = generate_answer(req.question, req.max_tokens, req.temperature)
#     return ChatResponse(
#         answer=answer,
#         tokens_used=tokens,
#         model="python-coding-assistant",
#         timestamp=datetime.now().isoformat()
#     )
#
# # ================= MAIN =================
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


