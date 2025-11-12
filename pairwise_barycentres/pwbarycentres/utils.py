import torch
from .data_processing import BarycentreDataProcessor, SinkhornDataProcessor


# ----------------------------------------------------------------------------------
#     ENTROPY RELATED THINGS
# ----------------------------------------------------------------------------------

def kl_prox(s, epsilon, rho, p):
    return s**(epsilon/(epsilon + rho)) * p**(rho/(epsilon + rho))

def balanced_entropy(f, epsilon, rho):
    return f

def kl_entropy(f, epsilon, rho, tol=1e-13):
    """
    KL entropy term: rho * f * (log(f) - 1)
    with the convention that 0 * log(0) = 0.
    Entries <= tol are treated as 0 for stability.
    """
    out = torch.zeros_like(f)
    mask = f > tol
    if mask.any():
        fm = f[mask]
        out[mask] = rho * (fm * (torch.log(fm) - 1.0))
    return out

def balanced_prox(s, epsilon, rho, p):
    return p

def kl_proxdiv(s, epsilon, rho, p, u=None):
    
    if u is None:
        return (p/s)**(rho/(epsilon+rho))
    return (p/s)**(rho/(epsilon+rho)) * torch.exp(-u/(epsilon + rho))

def balanced_proxdiv(s, epsilon, rho, p, u=None):
    return p/s

def tv_prox(s, epsilon, rho, p):
    return torch.min(s*torch.exp(rho/epsilon), torch.max(s*torch.exp(-rho/epsilon), p))

def tv_proxdiv(s, epsilon, rho, p, u=None):
    if u is None:
        u = 0.0
    return torch.min(torch.exp((rho - u)/epsilon), torch.max(torch.exp((-rho + u)/epsilon), p/s))

def chizat_proxdiv_step(s, epsilon, rho, p, aprox='kl', u=None):
    """
    u is for kernel truncation purposes which may be useful later
    """
    if aprox == 'kl':
        return kl_proxdiv(s, epsilon, rho, p, u)
    elif aprox == 'balanced':
        return balanced_proxdiv(s, epsilon, rho, p, u)
    elif aprox == 'tv':
        return tv_proxdiv(s, epsilon, rho, p, u)
    else:
        raise NotImplementedError("Only kl and balanced aprox implemented")

def _dual_cost_data_term(a, data, aprox, epsilon, rho):
    if aprox == 'kl':
        return - rho* torch.sum((a**(-epsilon/rho) - 1)*data)  # I'm worried we might get instabilities here
    elif aprox == 'balanced':
        return torch.sum(epsilon*torch.log(a) * data)
    elif aprox == 'tv':
        assert epsilon*torch.log(a) <= rho, "a should be less than rho for tv aprox"
        return torch.sum(rho * (torch.maximum(epsilon*torch.log(a), rho))* data)
    else:
        raise NotImplementedError("Only kl and balanced aprox implemented")

# ------------------------------------------------------------------------------------------------
# MISC
# ------------------------------------------------------------------------------------------------
def tensorise_f(C1, C2, f):
    """
    Perform the tensorised multiplication for regular cartesian grid

    Parameters
    ----------
    C1 : torch.Tensor
        X_x . Y_x (n1, m1)
    C2 : torch.Tensor
        X_y . Y_y (n2, m2)
    f : torch.Tensor
        weighting (n1, n2) or (m1, m2)

    Returns
    -------
    torch.Tensor
        output multiplication (m1, m2) or (n1, n2)
    """
    # ToDo: Create D dimensional verison
    # Check dimensions
    N, M = f.shape

    if N == C1.shape[0] and M == C2.shape[0]:
        ind = 0
    else:
        ind = 1

    return torch.tensordot(
        torch.tensordot(C1, f, dims=([ind], [0])), C2, dims=([1], [ind])
    )


def graph_creator_from_edges_and_weights(edges, weights=None):
    import networkx as nx

    if weights is None:
        weights = [1/len(edges) for _ in edges]

    G = nx.Graph()
    for i, edge in enumerate(edges):
        G.add_edge(edge[0], edge[1], weight=weights[i])
    return G


# 
def generate_barycentredataprocessor(data, barycentre_grid, grid=None, weights=None, cuda_device=None):
    # Data should be arranged as a list of lists
    # i.e. data[i] = [density, grid]
    # the grid and density could be [None, None]

    # barycentre can be equal to grid - it makes things more simple
    
    # Build graph
    M = len(data)
    edges = []
    counter = 0
    data_dict = {}
    for i in range(M):
        edges.append((counter, counter+1))
        data_dict[counter] = {
            'density': None, # this is the bayrcentre which will have a uniform density to start
            'grid': barycentre_grid,
        }
        data_dict[counter + 1] = {
            'density': data[i][0],
            'grid': grid if grid is not None else data[i][1],
        }
        counter += 2
    graph = graph_creator_from_edges_and_weights(edges, weights)

    # build data processor

    dp = SinkhornDataProcessor(
        potentials='a',
        data_dict=data_dict,
        graph=graph,
        free_grids=False,
        grid=grid,
        cuda_device=cuda_device,
    )

    # Put barycentres as the same grid data
    shared_density = data_dict[0]['density']
    for edge in dp.graph.edges:
        dp.data_dict[edge[0]]['density'] = shared_density

    # clean memory
    # We can't clean all memory we can only clean the grids that are not used by the data
    # we can also make sure the barycentre grid is pointing to the same grid.
    for edge in dp.graph.edges:
        if 'x1y1' in dp.data_dict[edge[1]]:
            del dp.data_dict[edge[1]]['grid']
    
    return dp
        