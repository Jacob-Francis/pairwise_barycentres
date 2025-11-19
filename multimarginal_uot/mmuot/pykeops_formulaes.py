import pykeops.torch as pykeops
import torch

_alpha_reduction_pykeops = pykeops.Genred(
    f"((F - IntInv(2)*SqDist(X, Y))/E)",
    [
        "F = Vi(1)",  # potential i
        "X = Vi(2)",  # grid i
        "Y = Vj(2)",  # grid j
        "E = Pm(1)",  # epsilon
        "M = Vi(1)",  # previous alpha(j,k) reductions
    ],
    reduction_op="Max_SumShiftExpWeight",
    axis=0,
    formula2="M", # its not in the formula but through weighted it dealt with
)

def alpha_reduction_pykeops(Fi, Xi, Yj, E, Mi):
    """
    returns Vj (1) and uses max_sumshiftexpweight (but we join them back)
    "F = Vi(1)",  # potential i
    f"X = Vi(2)",  # grid i
    f"Y = Vj(2)",  # grid j
    "E = Pm(1)",  # epsilon
    "M = Vi(1)",  # previous alpha(j,k) reductions
    """

    temp = _alpha_reduction_pykeops(Fi, Xi, Yj, E, Mi)

    return (torch.exp(temp[:, 0]) * temp[:, 1]).view(-1, 1)
 