from fastapi import FastAPI
from app.routes import users, ollama

app = FastAPI()
app.include_router(users.router)
app.include_router(ollama.router)

@app.get("/")
def home():
    return {"message": "FastAPI is running!"}

