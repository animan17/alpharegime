

import torch 
from alphagen.data.calculator import AlphaCalculator

class Alpha:
    def __init__(
        self,
        calculator: AlphaCalculator,
        device: torch.device = torch.device("cpu")
    ):
        self.calculator = calculator
        self.device = device
        self.ic_ret = 0
        self.expr: Expression

    def evaluate(self, expr: Expression)-> Tuple[float, float]:
        self.ic_ret = self.calculator.calc_single_IC_ret(expr)
        self.expr = expr    
        return self.ic_ret
    
    def test(self, calculator) -> Tuple[float, float]:
        return calculator.calc_single_IC_ret(self.expr)
