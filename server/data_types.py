from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date

class BankDataInput(BaseModel):
    bank_name: str
    date: str
    stock_price: float
    digital_transactions: int
    it_spending: float
    mobile_users: int

class AnalysisRequest(BaseModel):
    bank_names: List[str]
    metrics: Optional[List[str]] = None
    start_date: Optional[str] = None  
    end_date: Optional[str] = None