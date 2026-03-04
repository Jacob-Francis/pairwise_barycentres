from pykeops.torch import generic_logsumexp, generic_sum, Genred
import torch

_pykeops_chizat_reduction = generic_sum(
    f"(Exp((- IntInv(2)*SqDist(X, Y))/E)*S )",
    "f = Vj(1)",  # Geo: 1 scalar per line
    f"X = Vi(2)",  # Geo: 2-dim
    f"Y = Vj(2)",  # Uni: 1 scalar per line
    "E = Pm(1)",  # parameter: 1 scalar per line
    "S = Vi(1)",
)


def chizat_reduction(Xi, Yj, epsilon, ai):
    """

    "f = Vj(1)",  # Geo: 1 scalar per line
    f"X = Vi(2)",  # Geo: 2-dim
    f"Y = Vj(2)",  # Uni: 1 scalar per line
    "E = Pm(1)",  # parameter: 1 scalar per line
    "S = Vi(1)",


    """

    return _pykeops_chizat_reduction(
        Xi,
        Yj,
        epsilon.view(-1,1),
        ai,
    )


#  self.log_sum_exp_max_shift_weight = Genred(
#             f"((F + G - IntInv(2)*C*{self.cost_string})/E)",
#             [
#                 "G = Vj(1)",  # Uni: 1 scalar per line
#                 "F = Vi(1)",  # Geo: 1 scalar per line
#                 "X = Vi(2)",  # Geo: 2-dim
#                 "Y = Vj(2)",  # Uni: 1 scalar per line
#                 "E = Pm(1)",  # parameter: 1 scalar per line
#                 "C = Pm(1)",
#                 "M = Vj(1)",
#             ],
#             reduction_op="Max_SumShiftExpWeight",
#             axis=1,
#             formula2="M",
#         )

_pykeops_log_reduction_ii = Genred(
    f"((F - IntInv(2)*SqDist(X, Y))/E)",
    # "f = Vj(1)",  # Geo: 1 scalar per line
    ["F = Vi(1)",  # Geo: 1 scalar per line
    f"X = Vi(2)",  # Geo: 2-dim
    f"Y = Vj(2)",  # Uni: 1 scalar per line
    "E = Pm(1)",  # parameter: 1 scalar per line
    "S = Vi(1)"],
    reduction_op="Max_SumShiftExpWeight",
    axis=0,
    formula2="S",
)


def log_reduction_ii(Fi, Xi, Yj, epsilon, ai):
    """
    "f = Vj(1)",  # Geo: 1 scalar per line
    "F = Vi(1)",  # Geo: 1 scalar per line
    f"X = Vi(2)",  # Geo: 2-dim
    f"Y = Vj(2)",  # Uni: 1 scalar per line
    "E = Pm(1)",  # parameter: 1 scalar per line
    "S = Vi(1)",


    """

    temp =  _pykeops_log_reduction_ii(
        Fi,
        Xi,
        Yj,
        epsilon.view(-1,1),
        ai,
    )

    temp = temp[:, 0] + torch.log(temp[:, 1])
    return temp.view(-1, 1) 

_pykeops_log_reduction_ij = Genred(
    f"((F - IntInv(2)*SqDist(X, Y))/E)",
    # "f = Vj(1)",  # Geo: 1 scalar per line
    ["F = Vi(1)",  # Geo: 1 scalar per line
    f"X = Vi(2)",  # Geo: 2-dim
    f"Y = Vj(2)",  # Uni: 1 scalar per line
    "E = Pm(1)",  # parameter: 1 scalar per line
    "S = Vj(1)"],
    reduction_op="Max_SumShiftExpWeight",
    axis=0,
    formula2="S",
)

def log_reduction_ij(Fi, Xi, Yj, epsilon, aj):
    """

    "f = Vj(1)",  # Geo: 1 scalar per line
    f"X = Vi(2)",  # Geo: 2-dim
    f"Y = Vj(2)",  # Uni: 1 scalar per line
    "E = Pm(1)",  # parameter: 1 scalar per line
    "S = Vi(1)",


    """

    temp =  _pykeops_log_reduction_ij(
        Fi,
        Xi,
        Yj,
        epsilon.view(-1,1),
        aj,
    )

    temp = temp[:, 0] + torch.log(temp[:, 1])
    return temp.view(-1, 1) 

_pykeops_chizat_marginals = generic_sum(
    f"(Exp(( - IntInv(2)*SqDist(X, Y))/E)*S*P )",
    "f = Vj(1)",  # Geo: 1 scalar per line
    f"X = Vi(2)",  # Geo: 2-dim
    f"Y = Vj(2)",  # Uni: 1 scalar per line
    "E = Pm(1)",  # parameter: 1 scalar per line
    "S = Vi(1)",
    "P = Vj(1)",
    "C = Pm(1)",
)


def chizat_marginals(Xi, Yj, epsilon, ai, bj):
    """

    "f = Vj(1)",  # Geo: 1 scalar per line
    f"X = Vi(2)",  # Geo: 2-dim
    f"Y = Vj(2)",  # Uni: 1 scalar per line
    "E = Pm(1)",  # parameter: 1 scalar per line
    "S = Vi(1)",
    "P = Vj(1)",
    """

    return _pykeops_chizat_marginals(
        Xi,
        Yj,
        epsilon.view(-1,1),
        ai,
        bj,
    )




_pykeops_fg_reduction_ii = Genred(
    f"((F + G - IntInv(2)*SqDist(X, Y))/E)",
    ["F = Vi(1)", 
     "G = Vj(1)", 
    f"X = Vi(2)", 
    f"Y = Vj(2)",  
    "E = Pm(1)",
    "S = Vi(1)"],
    reduction_op="Max_SumShiftExpWeight",
    axis=0,
    formula2="S",
)

_pykeops_fg_reduction_ij = Genred(
    f"((F + G - IntInv(2)*SqDist(X, Y))/E)",
    ["F = Vi(1)", 
     "G = Vj(1)", 
    f"X = Vi(2)", 
    f"Y = Vj(2)",  
    "E = Pm(1)",
    "S = Vj(1)"],
    reduction_op="Max_SumShiftExpWeight",
    axis=0,
    formula2="S",
)


def fg_reduction_ii(Fi, Gj, Xi, Yj, epsilon, ai):
    """
    "f = Vj(1)",  # Geo: 1 scalar per line
    "F = Vi(1)",  # Geo: 1 scalar per line
    f"X = Vi(2)",  # Geo: 2-dim
    f"Y = Vj(2)",  # Uni: 1 scalar per line
    "E = Pm(1)",  # parameter: 1 scalar per line
    "S = Vi(1)",


    """
    if ai is None:
        ai = torch.ones_like(Fi)

    temp =  _pykeops_fg_reduction_ii(
        Fi,
        Gj,
        Xi,
        Yj,
        epsilon.view(-1,1),
        ai,
    )

    temp = torch.exp(temp[:, 0])*temp[:, 1]
    return temp.view(-1, 1) 

def fg_reduction_ij(Fi, Gj, Xi, Yj, epsilon, aj):
    """
    "f = Vj(1)",  # Geo: 1 scalar per line
    "F = Vi(1)",  # Geo: 1 scalar per line
    f"X = Vi(2)",  # Geo: 2-dim
    f"Y = Vj(2)",  # Uni: 1 scalar per line
    "E = Pm(1)",  # parameter: 1 scalar per line
    "S = Vi(1)",


    """
    if aj is None:
        aj = torch.ones_like(Gj)

    temp =  _pykeops_fg_reduction_ij(
        Fi,
        Gj,
        Xi,
        Yj,
        epsilon.view(-1,1),
        aj,
    )

    temp = torch.exp(temp[:, 0])*temp[:, 1]
    return temp.view(-1, 1) 



_pykeops_c_pi_term = Genred(
    f"((F + G - IntInv(2)*SqDist(X, Y))/E)",
    ["F = Vi(1)", 
     "G = Vj(1)", 
    f"X = Vi(2)", 
    f"Y = Vj(2)",  
    "E = Pm(1)",
    ],
    reduction_op="Max_SumShiftExpWeight",
    axis=0,
    formula2="IntInv(2)*SqDist(X, Y)",
)


def c_pi_term(Fi, Gj, Xi, Yj, epsilon):
    """
    "f = Vj(1)",  # Geo: 1 scalar per line
    "F = Vi(1)",  # Geo: 1 scalar per line
    f"X = Vi(2)",  # Geo: 2-dim
    f"Y = Vj(2)",  # Uni: 1 scalar per line
    "E = Pm(1)",  # parameter: 1 scalar per line


    """
    temp =  _pykeops_c_pi_term(
        Fi,
        Gj,
        Xi,
        Yj,
        epsilon.view(-1,1),
    )

    temp = torch.exp(temp[:, 0])*temp[:, 1]

    return temp.sum()

if __name__ == "__main__":
    pass
