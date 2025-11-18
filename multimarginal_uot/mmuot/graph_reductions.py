from .pykeops_formulaes import alpha_reduction_pykeops
import torch
from graph_dp import SinkhornDataProcessor
from .utils import aprox_lse_update




# process the dictionary to have an alpha variable on each edge (and in both directions)
# but for tensorisation we don't need to store in both directions
# As part of the processing make sure that x1y1 and x2y2 are avaliable from both orderes of the edge .


def sinkhorn_update(dp: SinkhornDataProcessor, j, epsilon, rho, aprox='balanced', prod=True):
    """
    Usual sinkhorn reduction though now over many branches using the recursive reduction done before
    """
    
    alpha = torch.ones_like(dp.data_dict[j]['f'])

    for i in dp.graph.neighbors(j):
        alpha *= dp.data_dict["edges"][(j, i)]["alpha"]

    # Log-sum-exp
    if prod:
        temp = - epsilon * torch.log(alpha)
    else:
        temp = epsilon * torch.log(dp.data_dict[j]["data"]) - epsilon*torch.log(alpha)

    # pointwise aprox;
    temp = aprox_lse_update(temp, epsilon, rho, aprox=aprox)

    # psuedo convergence err
    err = torch.linalg.norm(
        temp.view(-1, 1) - dp.data_dict[j]["f"].view(-1, 1),
        ord=float("inf"),
    ) # noqa: E1102

    return temp, err