## FIX COMMENTS, replaced period logic with day, stock tensor
## BRAINSTORM IF ANY NEW OPERATOR SHOULD BE INTRODUCED FOR THE CLUSTERING VERSION. ALSO CSRANK IS REMOVED


from abc import ABCMeta, abstractmethod
from typing import List, Type, Union, Tuple

import torch
from torch import Tensor
from alphagen.utils.maybe import Maybe, some, none
from alphagen_qlib.stock_data import StockData, FeatureType


_ExprOrFloat = Union["Expression", float]
_DTimeOrInt = Union["DeltaTime", int]


class OutOfDataRangeError(IndexError):
    pass


class Expression(metaclass=ABCMeta):                                                    # this is base class, everything is an expression
                                                                                        # instantiating defines the expression formula, it can output raw data like OHLC data for a time range, or it apply operations on raw data
                                                                                        # calling evaluate() allows passing stock data and time range on which we apply the expression formula, the time range and data passed does not become a property of the expression object (expression object is only a formula, not an actual quantity), the same expression object is used to evaluate for different datasets and ranges
    @abstractmethod
    def evaluate(self, data: StockData, days: Tensor, stocks: Tensor) -> Tensor: ...

    def __repr__(self) -> str: return str(self)

    def __add__(self, other: _ExprOrFloat) -> "Add": return Add(self, other)            # __add__ defines how + operator works on Expression objects, __add__ is a special keyword
    def __radd__(self, other: float) -> "Add": return Add(other, self)
    def __sub__(self, other: _ExprOrFloat) -> "Sub": return Sub(self, other)
    def __rsub__(self, other: float) -> "Sub": return Sub(other, self)
    def __mul__(self, other: _ExprOrFloat) -> "Mul": return Mul(self, other)
    def __rmul__(self, other: float) -> "Mul": return Mul(other, self)
    def __truediv__(self, other: _ExprOrFloat) -> "Div": return Div(self, other)
    def __rtruediv__(self, other: float) -> "Div": return Div(other, self)
    def __pow__(self, other: _ExprOrFloat) -> "Pow": return Pow(self, other)
    def __rpow__(self, other: float) -> "Pow": return Pow(other, self)
    def __pos__(self) -> "Expression": return self
    def __neg__(self) -> "Sub": return Sub(0., self)
    def __abs__(self) -> "Abs": return Abs(self)

    @property
    @abstractmethod
    def is_featured(self) -> bool: ...


class Feature(Expression):                                                              # child of Expression class
    def __init__(self, feature: FeatureType) -> None:                                   # instantiatiating defines type of feature it is used
        self._feature = feature                                                         
                                                                                            ### FIX COMMENTS ON EXPRESSION IS NOT QUANTITY BUT FORMULA
    def evaluate(self, data: StockData, days: Tensor, stocks: Tensor) -> Tensor:         # evaluate basically slices required OHLC data from data object and returns it
        # Shift the requested days by the backtrack window
        shifted_days = days + data.max_backtrack_days
        
        # Validation
        if shifted_days.min() < 0 or shifted_days.max() >= data.data.shape[0]:
            raise OutOfDataRangeError()        
        
        return data.data[shifted_days, int(self._feature), stocks]
    
    def __str__(self) -> str: return '$' + self._feature.name.lower()                   # __str__ defines what str() or print() outputs on Feature objects

    @property
    def is_featured(self): return True


class Constant(Expression):
    def __init__(self, value: float) -> None:                   
        self.value = value                              

    def evaluate(self, data: StockData, days: Tensor, stocks: Tensor) -> Tensor:        # Return a tensor of the same shape as `days`, filled with the constant
        return torch.full(
            size=days.shape, 
            fill_value=self.value, 
            dtype=data.data.dtype, 
            device=data.data.device
        )
    def __str__(self) -> str: return str(self.value)

    @property
    def is_featured(self): return False


class DeltaTime(Expression):
    # This is not something that should be in the final expression
    # It is only here for simplicity in the implementation of the tree builder
    def __init__(self, delta_time: int) -> None:                                        # only sets self.delta_time, evaluate() is never called
        self._delta_time = delta_time

    def evaluate(self, data: StockData, days: Tensor, stocks: Tensor) -> Tensor:
        assert False, "Should not call evaluate on delta time"

    def __str__(self) -> str: return f"{self._delta_time}d"

    @property
    def is_featured(self): return False


def _into_expr(value: _ExprOrFloat) -> "Expression":
    return value if isinstance(value, Expression) else Constant(value)                  # returns Expression object by converting scalar by instantiating Constant object


def _into_delta_time(value: Union[int, DeltaTime]) -> DeltaTime:
    return value if isinstance(value, DeltaTime) else DeltaTime(value)                  # returns Expression object by converting scalar by instantiating DeltaTime object


# Operator base classes



class Operator(Expression):
    @classmethod                                                                        # @classmethod tells python to pass the Class as first argument (by default object is passed to self paremeter), don't confuse with object instance, it can be called like <class>.classmethod() or <object>.classmethod(), but either way first parameter is set to the class
                                                                                        # like calling Operator.n_args() will set cls to the Operator class
    @abstractmethod
    def n_args(cls) -> int: ...                                         

    @classmethod
    @abstractmethod
    def category_type(cls) -> Type["Operator"]: ...

    @classmethod
    @abstractmethod
    def validate_parameters(cls, *args) -> Maybe[str]: ...

    @classmethod
    def _check_arity(cls, *args) -> Maybe[str]:                                         # checks if correct number of operands
        arity = cls.n_args()
        if len(args) == arity:
            return none(str)
        else:
            return some(f"{cls.__name__} expects {arity} operand(s), but received {len(args)}")

    @classmethod                                                                        
    def _check_exprs_featured(cls, args: list) -> Maybe[str]:                           # checks for type of each arg in args
        any_is_featured: bool = False
        for i, arg in enumerate(args):
            if not isinstance(arg, (Expression, float)):
                return some(f"{arg} is not a valid expression")
            if isinstance(arg, DeltaTime):
                return some(f"{cls.__name__} expects a normal expression for operand {i + 1}, "
                            f"but got {arg} (a DeltaTime)")
            any_is_featured = any_is_featured or (isinstance(arg, Expression) and arg.is_featured)
        if not any_is_featured:
            if len(args) == 1:
                return some(f"{cls.__name__} expects a featured expression for its operand, "
                            f"but {args[0]} is not featured")
            else:
                return some(f"{cls.__name__} expects at least one featured expression for its operands, "
                            f"but none of {args} is featured")
        return none(str)                                                               

    @classmethod
    def _check_delta_time(cls, arg) -> Maybe[str]:
        if not isinstance(arg, (DeltaTime, int)):
            return some(f"{cls.__name__} expects a DeltaTime as its last operand, but {arg} is not")
        return none(str)

    @property
    @abstractmethod
    def operands(self) -> Tuple[Expression, ...]: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({','.join(str(op) for op in self.operands)})"


class UnaryOperator(Operator):
    def __init__(self, operand: _ExprOrFloat) -> None:
        self._operand = _into_expr(operand)

    @classmethod
    def n_args(cls) -> int: return 1

    @classmethod
    def category_type(cls): return UnaryOperator

    @classmethod
    def validate_parameters(cls, *args) -> Maybe[str]:
        return cls._check_arity(*args).or_else(lambda: cls._check_exprs_featured([args[0]]))

    def evaluate(self, data: StockData, days: Tensor, stocks: Tensor) -> Tensor:
        return self._apply(self._operand.evaluate(data, days, stocks))                                # The evaluate() method defines the computation structure for this operator type
                                                                                                # (e.g., unary vs binary), while the specific operation logic for each subclass
                                                                                                # is implemented in _apply(). This avoids redundancy by separating structure from behavior.

    @abstractmethod
    def _apply(self, operand: Tensor) -> Tensor: ...

    @property
    def operands(self): return self._operand,

    @property
    def is_featured(self): return self._operand.is_featured


class BinaryOperator(Operator):
    def __init__(self, lhs: _ExprOrFloat, rhs: _ExprOrFloat) -> None:
        self._lhs = _into_expr(lhs)
        self._rhs = _into_expr(rhs)

    @classmethod
    def n_args(cls) -> int: return 2

    @classmethod
    def category_type(cls): return BinaryOperator

    @classmethod
    def validate_parameters(cls, *args) -> Maybe[str]:
        return cls._check_arity(*args).or_else(lambda: cls._check_exprs_featured([args[0], args[1]]))

    def evaluate(self, data: StockData, days: Tensor, stocks: Tensor) -> Tensor:
        return self._apply(self._lhs.evaluate(data, days, stocks), self._rhs.evaluate(data, days, stocks))

    @abstractmethod
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: ...

    def __str__(self) -> str: return f"{type(self).__name__}({self._lhs},{self._rhs})"

    @property
    def operands(self): return self._lhs, self._rhs

    @property
    def is_featured(self): return self._lhs.is_featured or self._rhs.is_featured


class RollingOperator(Operator):
    def __init__(self, operand: _ExprOrFloat, delta_time: _DTimeOrInt) -> None:
        self._operand = _into_expr(operand)
        self._delta_time = _into_delta_time(delta_time)._delta_time

    @classmethod
    def n_args(cls) -> int: return 2

    @classmethod
    def category_type(cls): return RollingOperator

    @classmethod
    def validate_parameters(cls, *args) -> Maybe[str]:
        return cls._check_arity(*args).or_else(
            lambda: cls._check_exprs_featured([args[0]])
        ).or_else(
            lambda: cls._check_delta_time(args[1])
        )

    def evaluate(self, data: StockData, days: Tensor, stocks: Tensor) -> Tensor:
        dt = self._delta_time
        # Create an offset array: e.g., for dt=3, offsets = [-2, -1, 0]
        offsets = torch.arange(-dt + 1, 1, dtype=days.dtype, device=days.device)
        
        # Broadcast days and stocks to include the rolling window dimension
        window_days = days.unsqueeze(-1) + offsets
        window_stocks = stocks.unsqueeze(-1).expand_as(window_days)
        
        # Evaluate operand. If input was [N], values will be [N, dt]
        values = self._operand.evaluate(data, window_days, window_stocks)
        
        # _apply functions (like mean, std, max) already operate on dim=-1
        return self._apply(values)
    

    @abstractmethod
    def _apply(self, operand: Tensor) -> Tensor: ...

    @property
    def operands(self): return self._operand, DeltaTime(self._delta_time)

    @property
    def is_featured(self): return self._operand.is_featured


class PairRollingOperator(Operator):                                                                # applying say covariance to stock i at t1 and stock j at t2, so we calculate same size past window for both and then apply operator which takes in both windows, like covariance operator
    def __init__(self, lhs: _ExprOrFloat, rhs: _ExprOrFloat, delta_time: _DTimeOrInt) -> None:
        self._lhs = _into_expr(lhs)
        self._rhs = _into_expr(rhs)
        self._delta_time = _into_delta_time(delta_time)._delta_time

    @classmethod
    def n_args(cls) -> int: return 3

    @classmethod
    def category_type(cls): return PairRollingOperator

    @classmethod
    def validate_parameters(cls, *args) -> Maybe[str]:
        return cls._check_arity(*args).or_else(
            lambda: cls._check_exprs_featured([args[0], args[1]])
        ).or_else(
            lambda: cls._check_delta_time(args[2])
        )

    def _unfold_one(self, expr: Expression, data: StockData, days: Tensor, stocks: Tensor) -> Tensor:
        dt = self._delta_time
        offsets = torch.arange(-dt + 1, 1, dtype=days.dtype, device=days.device)
        
        window_days = days.unsqueeze(-1) + offsets
        window_stocks = stocks.unsqueeze(-1).expand_as(window_days)
        
        return expr.evaluate(data, window_days, window_stocks)

    def evaluate(self, data: StockData, days: Tensor, stocks: Tensor) -> Tensor:
        lhs = self._unfold_one(self._lhs, data, days, stocks)
        rhs = self._unfold_one(self._rhs, data, days, stocks)
        return self._apply(lhs, rhs)


    @abstractmethod
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: ...

    @property
    def operands(self): return self._lhs, self._rhs, DeltaTime(self._delta_time)

    @property
    def is_featured(self): return self._lhs.is_featured or self._rhs.is_featured


# Operator implementations

class Abs(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.abs()


class Sign(UnaryOperator):
     def _apply(self, operand: Tensor) -> Tensor: return operand.sign()


class Log(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.log()



# class CSRank(UnaryOperator):                                                    
#     def _apply(self, operand: Tensor) -> Tensor:                                                # CHANGED 
#         nan_mask = operand.isnan()
#         filled = operand.masked_fill(nan_mask, float('inf'))

#         rank = filled.argsort(dim=1).argsort(dim=1).float()

#         n = (~nan_mask).sum(dim=1, keepdim=True)
#         rank = rank / n

#         rank[nan_mask] = torch.nan
#         return rank



class Add(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs + rhs


class Sub(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs - rhs


class Mul(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs * rhs


class Div(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs / rhs


class Pow(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs ** rhs


class Greater(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs.max(rhs)


class Less(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs.min(rhs)


class Ref(RollingOperator):
    # Ref is not *really* a rolling operator, in that other rolling operators
    # deal with the values in (-dt, 0], while Ref only deal with the values
    # at -dt. Nonetheless, it should be classified as rolling since it modifies
    # the time window.

    def evaluate(self, data: StockData, days: Tensor, stocks: Tensor) -> Tensor:
        # Shift the exact days requested
        shifted_days = days - self._delta_time
        return self._operand.evaluate(data, shifted_days, stocks)

    def _apply(self, operand: Tensor) -> Tensor:
        # This is just for fulfilling the RollingOperator interface
        ...


class Mean(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.mean(dim=-1)


class Sum(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.sum(dim=-1)


class Std(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.std(dim=-1)


class Var(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.var(dim=-1)


class Skew(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        # skew = m3 / m2^(3/2)
        central = operand - operand.mean(dim=-1, keepdim=True)
        m3 = (central ** 3).mean(dim=-1)
        m2 = (central ** 2).mean(dim=-1)
        return m3 / m2 ** 1.5


class Kurt(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        # kurt = m4 / var^2 - 3
        central = operand - operand.mean(dim=-1, keepdim=True)
        m4 = (central ** 4).mean(dim=-1)
        var = operand.var(dim=-1)
        return m4 / var ** 2 - 3


class Max(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.max(dim=-1)[0]


class Min(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.min(dim=-1)[0]


class Med(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.median(dim=-1)[0]


class Mad(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        central = operand - operand.mean(dim=-1, keepdim=True)
        return central.abs().mean(dim=-1)


class Rank(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        last = operand[:, :, -1, None]
        left = (last < operand).count_nonzero(dim=-1)
        right = (last <= operand).count_nonzero(dim=-1)
        result = (right + left + (right > left)) / (2 * n)
        return result


class Delta(RollingOperator):
    # Delta is not *really* a rolling operator, in that other rolling operators
    # deal with the values in (-dt, 0], while Delta only deal with the values
    # at -dt and 0. Nonetheless, it should be classified as rolling since it
    # modifies the time window.

    def evaluate(self, data: StockData, days: Tensor, stocks: Tensor) -> Tensor:
        # Evaluate current
        curr_values = self._operand.evaluate(data, days, stocks)
        
        # Evaluate past
        shifted_days = days - self._delta_time
        past_values = self._operand.evaluate(data, shifted_days, stocks)
        
        return curr_values - past_values
    


    def _apply(self, operand: Tensor) -> Tensor:
        # This is just for fulfilling the RollingOperator interface
        ...


class WMA(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        weights = torch.arange(n, dtype=operand.dtype, device=operand.device)
        weights /= weights.sum()
        return (weights * operand).sum(dim=-1)


class EMA(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        alpha = 1 - 2 / (1 + n)
        power = torch.arange(n, 0, -1, dtype=operand.dtype, device=operand.device)
        weights = alpha ** power
        weights /= weights.sum()
        return (weights * operand).sum(dim=-1)


class Cov(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        n = lhs.shape[-1]
        clhs = lhs - lhs.mean(dim=-1, keepdim=True)
        crhs = rhs - rhs.mean(dim=-1, keepdim=True)
        return (clhs * crhs).sum(dim=-1) / (n - 1)


class Corr(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        clhs = lhs - lhs.mean(dim=-1, keepdim=True)
        crhs = rhs - rhs.mean(dim=-1, keepdim=True)
        ncov = (clhs * crhs).sum(dim=-1)
        nlvar = (clhs ** 2).sum(dim=-1)
        nrvar = (crhs ** 2).sum(dim=-1)
        stdmul = (nlvar * nrvar).sqrt()
        stdmul[(nlvar < 1e-6) | (nrvar < 1e-6)] = 1
        return ncov / stdmul


Operators: List[Type[Operator]] = [
    # Unary
    Abs, Sign, Log, 
    # Binary
    Add, Sub, Mul, Div, Pow, Greater, Less,
    # Rolling
    Ref, Mean, Sum, Std, Var, Skew, Kurt, Max, Min,
    Med, Mad, Rank, Delta, WMA, EMA,
    # Pair rolling
    Cov, Corr
]
