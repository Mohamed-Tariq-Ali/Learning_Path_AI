from fastapi import FastAPI
from Auth.routes import router as Auth_router
from User.routes import router as users_router

app = FastAPI()

app.include_router(Auth_router, prefix="/auth", tags=["auth"])
app.include_router(users_router, prefix="/users", tags=["users"])

@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI authentication and authorization example"}