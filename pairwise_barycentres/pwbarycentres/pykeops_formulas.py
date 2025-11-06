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
    f"(Exp(( - IntInv(2)*C*SqDist(X, Y))/E)*S*P )",
    "f = Vj(1)",  # Geo: 1 scalar per line
    f"X = Vi(2)",  # Geo: 2-dim
    f"Y = Vj(2)",  # Uni: 1 scalar per line
    "E = Pm(1)",  # parameter: 1 scalar per line
    "S = Vi(1)",
    "P = Vj(1)",
    "C = Pm(1)",
)

def chizat_marginals(Xi, Yj, epsilon, ai, bj, cost_const):
    """
    
    "f = Vj(1)",  # Geo: 1 scalar per line
    f"X = Vi(2)",  # Geo: 2-dim
    f"Y = Vj(2)",  # Uni: 1 scalar per line
    "E = Pm(1)",  # parameter: 1 scalar per line
    "S = Vi(1)",
    "P = Vj(1)",
    "C = Pm(1)",


    """
    
    return _pykeops_chizat_marginals(
        Xi,
        Yj,
        epsilon,
        ai,
        bj,
        cost_const,
    )
