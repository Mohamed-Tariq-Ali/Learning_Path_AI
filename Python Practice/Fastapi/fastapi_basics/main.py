from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello FastAPI"}
@app.get("/ping")
def ping():
    return {"response":"pong"}
