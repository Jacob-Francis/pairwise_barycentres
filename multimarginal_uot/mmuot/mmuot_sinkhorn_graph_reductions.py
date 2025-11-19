from .pykeops_formulaes import alpha_reduction_pykeops
import torch
from graph_dp import SinkhornDataProcessor
from .utils import aprox_lse_update, alpha_reduction
import networkx as nx

def generate_dp():
    pass

def procesinng_data_holder_for_mmot():
    pass
# process the dictionary to have an alpha variable on each edge (and in both directions)
# but for tensorisation we don't need to store in both directions
# As part of the processing make sure that x1y1 and x2y2 are avaliable from both orderes of the edge .


def sinkhorn_update(dp: SinkhornDataProcessor, j, epsilon, rho, aprox='balanced', prod=True):
    """
    Usual sinkhorn reduction though now over many branches using the recursive reduction done before
    """
    
    temp = torch.ones_like(dp.data_dict[j]['f'])

    # ToDo i think i shoudl be able to stablaise this since we have the max weighted shift inside 
    # pykeops, and then we take log
    for i in dp.graph.neighbors(j):
        temp *= dp.data_dict["edges"][(j, i)]["alpha"]

    # Log-sum-exp
    if prod:
        temp = - epsilon * torch.log(temp)
    else:
        temp = epsilon * torch.log(dp.data_dict[j]["data"]) - epsilon*torch.log(temp)

    # pointwise aprox;
    temp = aprox_lse_update(temp, epsilon, rho, aprox=aprox)

    # psuedo convergence err
    err = torch.linalg.norm(
        temp.view(-1, 1) - dp.data_dict[j]["f"].view(-1, 1),
        ord=float("inf"),
    ).item() # noqa: E1102

    return temp, err

def mmuot_sinkhorn_loop(dp: SinkhornDataProcessor,
    epsilon,
    rho,
    max_iterations=20,
    tol=1e-7,
    aprox='balanced',
    prod=True,
    convergence_tracking=False,
    verbose=False,
    ):

    # Initialisations
    err = tol + 1.
    count = 0

    if convergence_tracking:
        convergence_tracking = []

    # root node
    v0 = list(dp.graph.nodes)[0]

    # Depth first traversal to do reductions
    dfs_edges = list(nx.dfs_tree(dp.graph, source=v0).edges)
    
    # checking order
    print("DFS edges order: ", dfs_edges)
    print("Reversed DFS edges order: ", list(reversed(dfs_edges)), 'can j appear more than once?')

    # First initialisation of alphas'
    # ToDo: do i save internally inside alpha reduction?
    for p_j, j in reversed(dfs_edges):
        dp.data_dict[(p_j, j)]['alpha'] = alpha_reduction(dp, p_j, j, epsilon, prod=prod)
    
    # Begin Sinkhorn loop:
    while count < max_iterations and err > tol:
        err = -torch.inf
        
        for p_j, j in dfs_edges:
            # update alpha
            if j != v0:
                dp.data_dict[(j, p_j)]['alpha'] = alpha_reduction(dp, j, p_j, epsilon, prod=prod)
            # Sinkhorn update (including aprox relaxation)
            dp.data_dict[j]['f'], er = sinkhorn_update(dp, j, epsilon, rho, aprox=aprox)

            err = max(err, er)
        
        # update reverse alpha
        for p_j, j in reversed(dfs_edges):
            dp.data_dict[(p_j, j)]['alpha'] = alpha_reduction(dp, p_j, j, epsilon, prod=prod)
        
        count += 1

        if convergence_tracking:
            convergence_tracking.append(err.item())
        
        if verbose:
            print(f"Iteration {count}, Error: {err}")

    return dp

def mmuot_marginal_j(dp: SinkhornDataProcessor, j, epsilon, prod=True, update_alpha=False):
    
    marginal = torch.ones_like(dp.data_dict[j]['f'])

    # Gather incoming alphas
    for i in dp.graph.neighbors(j):
        if update_alpha:
            alpha_reduction(dp, j, i, epsilon, prod=prod)
        marginal *= dp.data_dict[(j, i)]['alpha']
    
    if prod:
        marginal = torch.exp(-dp.data_dict[j]['f']/epsilon) * marginal * dp.data_dict[j]['density']
    else:
        marginal = torch.exp((dp.data_dict[j]['f'])/epsilon) * marginal
    
    # marginal errror
    err = torch.linalg.norm(
        marginal.view(-1) - dp.data_dict[j]['data'].view(-1), ord=float("inf")
    ).item()

    return marginal, err

def mmuot_marginals(dp: SinkhornDataProcessor, epsilon, prod=True):
    
    marginals = {}
    errors = {}

    for j in dp.graph.nodes:
        marginals[j], errors[j] = mmuot_marginal_j(dp, j, epsilon, prod=prod)
    
    return marginals, errors


def mmuot_dual_cost(dp: SinkhornDataProcessor, epsilon, rho, aprox='balanced', prod=True, debiasing=False):
    
    cost = 0.0

    return cost