from fastapi import FastAPI
from typing import Literal, Optional
from pydantic import BaseModel

class Ticket(BaseModel):
    id: int
    priority: Literal["low", "medium", "high"]
    status: Literal["open", "in-progress", "on-hold", "closed"]
    is_gdpr: Optional[bool] = False
    
app = FastAPI()

@app.post("/ticket/")
def create_user(ticket: Ticket):
    return ticket