from pykeops.torch import generic_sum

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
        epsilon,
        ai,
    )


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
        epsilon,
        ai,
        bj,
    )


if __name__ == "__main__":
    # test chizat_reduction
    import torch
    Xi = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    Yj = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    epsilon = torch.tensor([0.1])
    ai = torch.tensor([[1.0], [2.0]])

    result = chizat_reduction(Xi, Yj, epsilon, ai)
    print("Chizat reduction result:", result)