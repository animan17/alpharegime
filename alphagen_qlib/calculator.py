from typing import Optional
from torch import Tensor
from alphagen.data.calculator import TensorAlphaCalculator
from alphagen.data.expression import Expression
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen_qlib.stock_data import StockData


class QLibStockDataCalculator(TensorAlphaCalculator):
    def __init__(self, data: StockData, days: Tensor, stocks: Tensor, target: Optional[Expression] = None):
        super().__init__(target.evaluate(data, days, stocks) if target is not None else None)    ## Removed normalize_by_day temporarily for simplicity in clustering case
        self.data = data
        self.days = days
        self.stocks = stocks

    def evaluate_alpha(self, expr: Expression) -> Tensor:
        return expr.evaluate(self.data, self.days, self.stocks)                               ## Removed normalize_by_day temporarily for simplicity in clustering case
    
    @property
    def n_days(self) -> int:
        return self.data.n_days
