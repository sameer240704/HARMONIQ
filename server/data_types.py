from typing import Optional, List
from pydantic import BaseModel

class BankDataInput(BaseModel):
    bank_name: str
    date: str
    stock_price: Optional[float] = None
    digital_transactions: Optional[int] = None
    it_spending: Optional[float] = None
    mobile_users: Optional[int] = None

class AnalysisRequest(BaseModel):
    bank_names: List[str]
    start_date: str
    end_date: str
    metrics: List[str]