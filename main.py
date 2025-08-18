from fastapi import FastAPI , WebSocket , WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from routers.router import router
import asyncio
from dotenv import load_dotenv
# from graph import app 
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from tools.state_management import app


api = FastAPI()

# Configure CORS
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


api.include_router(router)

@api.get("/")
def read_root():
    return {"message": "Backend is running!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)