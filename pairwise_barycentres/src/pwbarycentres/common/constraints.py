from .utils import Kd_dual_potential_reduction
from .marginals  import calculate_node_marginal, ot_marginals
import torch

def check_tensor_dtypes(obj, path=""):
    """
    Recursively walk through nested dict/list/tuple/objects
    and print any torch.Tensor that is NOT float64.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            check_tensor_dtypes(v, path + f".{k}")

    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            check_tensor_dtypes(v, path + f"[{i}]")

    elif isinstance(obj, torch.Tensor):
        if obj.dtype != torch.float64:
            print(f"[WARNING] Tensor at {path} has dtype {obj.dtype}, not float64")

    # ignore everything else (ints, floats, strings, None, etc.)


def d_constraint(d, epsilon, dp):
    """ 
    d = sum_k pi_k .T 1 / Kd
    """

    # pik_1 is the marginal on the barycentre
    sum_pi_k = torch.zeros_like(d)
    for node, e2, w in dp.graph.edges(data=True):
        mar, _ = calculate_node_marginal(dp, node, epsilon, debiasing=True)
        sum_pi_k += w['weight']*mar

    Kd = Kd_dual_potential_reduction(dp, d, epsilon)

    residual = d.view(-1,1) - sum_pi_k.view(-1,1)  / Kd.view(-1,1) 

    return torch.norm(residual, p=float('inf')), torch.norm(residual, p=1)

def barycentre_constraint(barycentre, epsilon, dp, debiasing=True):
    """ 
    0 = sum omega f_i = sum omega epsilon log(a_{bary})
    """

    # pik_1 is the marginal on the barycentre
    sumfi = torch.zeros_like(barycentre)
    for node, e2, w in dp.graph.edges(data=True):
        sumfi += w['weight'] * epsilon * torch.log(dp.data_dict[node]['a'] + 1e-30)

    return torch.norm(sumfi, p=float('inf')), torch.norm(sumfi, p=1)

