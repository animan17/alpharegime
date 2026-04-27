
#check, fully ai written!!!

from typing import Optional
from torch import Tensor
from alphagen.data.calculator import TensorAlphaCalculator
from alphagen.data.expression import Expression
from alphagen_qlib.stock_data import StockData

class QLibStockDataCalculator(TensorAlphaCalculator):
    def __init__(self, data: StockData, days: Tensor, stocks: Tensor, target: Optional[Expression] = None):
        self.data = data
        self.days = days
        self.stocks = stocks
        
        # Calculate the exact valid boundaries for this specific cluster
        self.min_day = int(self.days.min().item()) if len(self.days) > 0 else 0
        self.max_day = int(self.days.max().item() + 1) if len(self.days) > 0 else 0
        
        # 1. Evaluate TARGET densely ONLY within the safe cluster boundaries
        if target is not None and self.max_day > self.min_day:
            dense_target = target.evaluate(data, slice(self.min_day, self.max_day))
            
            # Offset the day indices because the dense matrix now starts at min_day instead of 0
            sparse_target = dense_target[self.days - self.min_day, self.stocks]
        else:
            sparse_target = torch.empty(0, device=days.device) if target is not None else None
            
        super().__init__(sparse_target)

    def evaluate_alpha(self, expr: Expression) -> Tensor:
        if self.max_day <= self.min_day:
            return torch.empty(0, device=self.days.device)
            
        # Evaluate RL ALPHA densely ONLY within the safe cluster boundaries
        dense_output = expr.evaluate(self.data, slice(self.min_day, self.max_day))
        
        # Offset the day indices to extract the correct pairs
        sparse_output = dense_output[self.days - self.min_day, self.stocks]
        
        return sparse_output
    
    @property
    def n_days(self) -> int:
        return self.data.n_days
