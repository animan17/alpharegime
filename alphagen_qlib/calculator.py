
#check, fully ai written!!!

from typing import Optional
import torch
from torch import Tensor
from alphagen.data.calculator import TensorAlphaCalculator
from alphagen.data.expression import Expression, Feature
from alphagen_qlib.stock_data import StockData, FeatureType

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


            # 2. EXTRACT SPARSE CLOSE PRICE FOR THE NEW EVALUATION MATH
            close_expr = Feature(FeatureType.CLOSE)
            dense_close = close_expr.evaluate(data, slice(self.min_day, self.max_day))
            self.sparse_close = dense_close[self.days - self.min_day, self.stocks]
        else:
            sparse_target = torch.empty(0, device=days.device) if target is not None else None
            self.sparse_close = torch.empty(0, device=days.device)
            
        super().__init__(sparse_target)

    def evaluate_alpha(self, expr: Expression) -> Tensor:
        if self.max_day <= self.min_day:
            return torch.empty(0, device=self.days.device)
            
        # Evaluate RL ALPHA densely ONLY within the safe cluster boundaries
        dense_output = expr.evaluate(self.data, slice(self.min_day, self.max_day))
        
        # Offset the day indices to extract the correct pairs
        sparse_output = dense_output[self.days - self.min_day, self.stocks]

        # 2. APPLY YOUR CUSTOM LOGIC: (Alpha / Current_Close) - 1
        transformed_alpha = (sparse_output / self.sparse_close) - 1
        
        return transformed_alpha
    
    @property
    def n_days(self) -> int:
        return self.data.n_days
