import torch
from graph_dp import SinkhornDataProcessor
from .pykeops_formulas import chizat_reduction, log_reduction_ii, log_reduction_ij

# ----------------------------------------------------------------------------------
#     ENTROPY RELATED THINGS
# ----------------------------------------------------------------------------------


def kl_prox(s, epsilon, rho, p):
    return s ** (epsilon / (epsilon + rho)) * p ** (rho / (epsilon + rho))


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
        return torch.where(p > 0, (p / s) ** (rho / (epsilon + rho)), torch.zeros_like(p))
        # return (p / s) ** (rho / (epsilon + rho))
    
    return torch.where(p > 0, (p / s) ** (rho / (epsilon + rho)), torch.zeros_like(p)) * torch.exp(-u / (epsilon + rho))


def balanced_proxdiv(s, epsilon, rho, p, u=None):

    return torch.where(p > 0, p / s, torch.zeros_like(p))


def tv_prox(s, epsilon, rho, p):
    return torch.min(
        s * torch.exp(rho / epsilon), torch.max(s * torch.exp(-rho / epsilon), p)
    )


def tv_proxdiv(s, epsilon, rho, p, u=None):
    if u is None:
        u = 0.0
    return torch.where(p > 0, torch.min(
        torch.exp((rho - u) / epsilon),
        torch.max(torch.exp((-rho - u) / epsilon), p / s),
    ), torch.zeros_like(p))
    # return torch.min(
    #     torch.exp((rho - u) / epsilon),
    #     torch.max(torch.exp((-rho - u) / epsilon), p / s),
    # )


def chizat_proxdiv_step(s, epsilon, rho, p, aprox="kl", u=None):
    """
    u is for kernel truncation purposes which may be useful later
    """
    if aprox == "kl":
        return kl_proxdiv(s, epsilon, rho, p, u)
    elif aprox == "balanced":
        return balanced_proxdiv(s, epsilon, rho, p, u)
    elif aprox == "tv":
        return tv_proxdiv(s, epsilon, rho, p, u)
    else:
        raise NotImplementedError("Only kl and balanced aprox implemented")


# def kl_log_aprox(s, epsilon, rho, p, zero_tol=1e-12):
#     """_summary_

#     Parameters
#     ----------
#     s : _type_
#         Reduction term`
#     epsilon : _type_
#         _description_
#     rho : _type_
#         _description_
#     p : _type_
#         data term 

#     Returns
#     -------
#     _type_
#         _description_
#     """
#     # ToD(o - do i return zero or s?
#     return  torch.where(p>zero_tol, (s - epsilon*torch.log(p)) * rho/(epsilon + rho), 1e3*torch.ones_like(s))

# def balanced_log_aprox(s, epsilon, rho, p, tol=1e-12):
#     return torch.where(p>tol, s - epsilon*torch.log(p), 1e3*torch.ones_like(s))

# def log_aprox_step(s, epsilon, rho, p, aprox="kl"):
#     """
#     u is for kernel truncation purposes which may be useful later
#     """
#     if aprox == "kl":
#         return kl_log_aprox(s, epsilon, rho, p)
#     elif aprox == "balanced":
#         return balanced_log_aprox(s, epsilon, rho, p)
#     elif aprox == "tv":
#         raise NotImplementedError("Only kl and balanced aprox implemented")
#     else:
#         raise NotImplementedError("Only kl and balanced aprox implemented")



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
    elif N == C1.shape[1] and M == C2.shape[1]:
        ind = 1
    else:
        raise ValueError(
            "Dimensions of C1, C2, and f do not match for tensorised multiplication, shapes are {}, {}, {}".format(
                C1.shape, C2.shape, f.shape
            )
        )

    return torch.tensordot(
        torch.tensordot(C1, f, dims=([ind], [0])), C2, dims=([1], [ind])
    )


def _tensorised_sinkhorn_reduction(a, x1y1, x2y2, epsilon, d=None, ind=None):

    # kernel computations - K @ a
    # main bottle neck
    # return tensorise_f(torch.exp(-x1y1 / epsilon), torch.exp(-x2y2 / epsilon), a)

    if ind==0:
        return tensorise_f(torch.exp((-x1y1) / epsilon), torch.exp((-x2y2) / epsilon), a*d)
    elif ind==1:
        return d*tensorise_f(torch.exp((-x1y1) / epsilon), torch.exp((-x2y2) / epsilon), a)
    else:
        return tensorise_f(torch.exp((-x1y1) / epsilon), torch.exp((-x2y2) / epsilon), a)


def _tensorised_log_sinkhorn_reduction(f, d, ind, x1y1, x2y2, epsilon):
    """
    f dual potential being reduced over
    d debiasing potential
    ind - 0 if d and f go together (and are summed), 1 if they are opposite so d is not summed. 
    """
    # kernel computations - K @ a
    # main bottle neck
    if ind==0:
        temp = tensorise_f(torch.exp((-x1y1) / epsilon), torch.exp((-x2y2) / epsilon), torch.exp(f / epsilon)*d)
    else:
        temp = d*tensorise_f(torch.exp((-x1y1) / epsilon), torch.exp((-x2y2) / epsilon), torch.exp(f / epsilon))

    return torch.log(temp)


def _tensorised_log_sinkhorn_reduction_stabalised(f, d, ind, x1y1, x2y2, epsilon):
    """
    f dual potential being reduced over
    d debiasing potential
    ind - 0 if d and f go together (and are summed), 1 if they are opposite so d is not summed. 
    """
    # kernel computations - K @ a
    # main bottle neck
    stabiliser = torch.max(f / epsilon)
    print("stabiliser", stabiliser)
    temp = torch.exp(f / epsilon - stabiliser)
    if torch.any(torch.isnan(temp)) or torch.any(torch.isinf(temp)):
        raise ValueError("NaN/inf detected in exp stabiliser", temp.min().item(), temp.max().item(), temp.sum().item(), stabiliser.item())

    if ind==0:
        temp = tensorise_f(torch.exp((-x1y1) / epsilon), torch.exp((-x2y2) / epsilon), torch.exp(f / epsilon - stabiliser)*d)
    else:
        temp = d*tensorise_f(torch.exp((-x1y1) / epsilon), torch.exp((-x2y2) / epsilon), torch.exp(f / epsilon - stabiliser))

    s = torch.log(temp) + stabiliser

    return torch.log(temp) + stabiliser


def graph_creator_from_edges_and_weights(edges, weights=None):
    import networkx as nx

    if weights is None:
        weights = [1 / len(edges) for _ in edges]

    G = nx.Graph()
    for i, edge in enumerate(edges):
        G.add_edge(edge[0], edge[1], weight=weights[i])
    return G


#
def generate_barycentredataprocessor(
    data, barycentre_grid, 
    grid=None, 
    weights=None, 
    cuda_device=None, 
    potentials="a",
    force_pykeops=False
):
    """ # Data should be arranged as a list of lists
        # i.e. data[i] = [density, grid]
        # the grid and density could be [None, None]

        # barycentre can be equal to grid - it makes things more simple
    """
    # Build graph
    M = len(data)
    edges = []
    counter = 0
    data_dict = {}

    if force_pykeops:
        if (grid is not None) and isinstance(grid, tuple):
            grid = torch.cartesian_prod(*grid)
        if isinstance(barycentre_grid, tuple):
            barycentre_grid = torch.cartesian_prod(*[torch.tensor(d) for d in barycentre_grid])
        else: 
            raise ValueError("If force_pykeops is True, barycentre_grid should be a tuple of 1D tensors to cartesian product")

    for i in range(M):
        edges.append((counter, counter + 1))
        data_dict[counter] = {
            "density": None,  # this is the bayrcentre which will have a uniform density to start
            "grid": barycentre_grid,
        }
        if grid is not None:
            data_dict[counter + 1] = {
                "density": data[i][0] if not force_pykeops else data[i][0].reshape(-1),
                "grid": grid if grid is not None else data[i][1],
            }
        else:
            if force_pykeops:
                tempgrid = torch.cartesian_prod(*[torch.tensor(d) for d in data[i][1]])
                data_dict[counter + 1] = {
                    "density": data[i][0] if not force_pykeops else data[i][0].reshape(-1),
                    "grid": tempgrid
                }
            else:
                data_dict[counter + 1] = {
                    "density": data[i][0],
                    "grid": data[i][1],
                }
        counter += 2
    graph = graph_creator_from_edges_and_weights(edges, weights)

    # build data processor

    dp = SinkhornDataProcessor(
        potentials=potentials,
        data_dict=data_dict,
        graph=graph,
        free_grids=False,
        cuda_device=cuda_device,
    )

    # Put barycentres as the same grid data
    shared_density = data_dict[0]["density"]
    for edge in dp.graph.edges:
        dp.data_dict[edge[0]]["density"] = shared_density

    # clean memory
    # We can't clean all memory we can only clean the grids that are not used by the data
    # we can also make sure the barycentre grid is pointing to the same grid.
    for edge in dp.graph.edges:
        if "x1y1" in dp.data_dict[edge[1]]:
            del dp.data_dict[edge[1]]["grid"]

    return dp

def generate_epsilon_list(epsilon_end: float, max_iter=1000):
    t = torch.arange(1, max_iter + 1, dtype=torch.float64)
    eps = 0.5 / torch.sqrt(t).to(epsilon_end)

    # clamp 
    eps = torch.clamp(eps, min=epsilon_end)
    return eps

def process_dict_for_barycentre(dp: SinkhornDataProcessor, debiasing=True):
    """
    Ensure that the barycentre nodes have the same density and a potential
    """

    # ToDo: check dp set up correctly
    # I suppose we could actually slove the problem thorugh different grids in which case they'd
    # have different grids but lets leave that for now
    for edge1 in dp.graph.edges:
        for edge2 in dp.graph.edges:
            assert (
                dp.data_dict[edge1[0]]["density"] is dp.data_dict[edge2[0]]["density"]
            ), "Barycentre node should have the same data"

    if debiasing:
        edge2 = list(dp.graph.edges)[0]

        # Need to add x1x1 x2x2 for debiasing potential term
        if "x1y1" in dp.data_dict[edge2] and "x2y2" in dp.data_dict[edge2]:
            # we can tensorise, so we must be able to tensorize the symmetric problem

            # Does eveyone have the same grid?
            # assert edge1 != edge2, "{edge1}, {edge2} should be different edges"
            if dp.data_dict[edge1]["x1y1"] is dp.data_dict[edge2]["x1y1"]:
                # then the edges are sharing a grid
                # So assign to the firs edge barycentre node the x1x1, x2x2
                dp.data_dict[edge2[0]]["x1x1"] = dp.data_dict[edge2]["x1y1"]
                dp.data_dict[edge2[0]]["x2x2"] = dp.data_dict[edge2]["x2y2"]
            else:
                # Not eveyone shares the same grid so need to compute symmetric version
                # it should still have a grid associated
                grid = dp.data_dict[edge2[0]]["grid"]
                if isinstance(grid, tuple):
                    dp.data_dict[edge2[0]]["x1x1"], dp.data_dict[edge2[0]]["x2x2"] = (
                        dp._cost_for_tuple(grid, grid)
                    )

                elif len(grid.shape) == 3 and len(grid.shape) == 3:
                    n1, n2, n3 = grid.shape
                    assert n3 == 2, "We assume 2D points"

                    # Calculate cost matrices - the indexing works
                    # because torch cdist eliminats the common axis which will have the same values.
                    dp.data_dict[edge2[0]]["x1x1"], dp.data_dict[edge2[0]]["x2x2"] = (
                        dp._cost_for_meshgrid(grid, grid, n1, n2, n1, n2)
                    )

            # point all barycentres to the tensorisation
            for edges in dp.graph.edges:
                dp.data_dict[edges[0]]["x1x1"] = dp.data_dict[edge2[0]]["x1x1"]
                dp.data_dict[edges[0]]["x2x2"] = dp.data_dict[edge2[0]]["x2x2"]

                assert (
                    dp.data_dict[edges[0]]["x1x1"] is dp.data_dict[edge2[0]]["x1x1"]
                ), "Barycentre nodes should share the same x1x1"
                assert (
                    dp.data_dict[edges[0]]["x2x2"] is dp.data_dict[edge2[0]]["x2x2"]
                ), "Barycentre nodes should share the same x2x2"

        elif "grid" in dp.data_dict[edge2[0]]:
            pass  # we can use PyKeOps

def _flat_grid_sinkhorn_reduction(a, X, Y, epsilon, d=None, ind=None):

    # # wheres the tuple!!!
    # print('a', type(a), a.shape)
    # print('X', type(X), X.shape)
    # print('Y', type(Y), Y.shape)
    # print('epsilon', type(epsilon), epsilon)
    # print('d', type(d), d.shape if d is not None else None)

    # kernel computations - K @ a
    # main bottle neck

    if ind==0:
        return chizat_reduction(X, Y, epsilon, a*d)
    elif ind==1:
        return d*chizat_reduction(X, Y, epsilon, a)
    else:
        return chizat_reduction(X, Y, epsilon, a)

def _dual_cost_data_term(a, data, aprox, epsilon, rho):
    '''
    Handles the double negative inside here! Don't add another
    '''

    assert a.shape == data.shape, "Shapes of a and data should match"

    if aprox == "kl":
        return torch.sum(torch.where(data > 0, -rho * (
            (a ** (-epsilon / rho) - 1) * data
        ) ,  torch.zeros_like(data)))
    elif aprox == "balanced":
        return torch.sum(torch.where(data > 0, (epsilon * torch.log(a) * data),  torch.zeros_like(data)))
    elif aprox == "tv":
        assert (epsilon * torch.log(torch.where(data > 0, a, torch.ones_like(a))) <= rho).all(), "a should be less than rho for tv aprox"
        return torch.sum(torch.where(data > 0, (-torch.maximum(-epsilon * torch.log(a), -rho)) * data, torch.zeros_like(data)))
    else:
        raise NotImplementedError("Only kl and balanced aprox implemented")

def _dual_cost_data_term_f_potential(f, data, aprox, epsilon, rho, zero_tol=1e-40):
    '''
    Handles the double negative inside here! Don't add another
    '''

    assert f.shape == data.shape, "Shapes of f and data should match"
    
    if aprox == "kl":
        return torch.where(data >= zero_tol, -rho * (torch.exp(-f / rho) - 1) * data, torch.zeros_like(data)).sum()
    elif aprox == "balanced":
        return torch.sum(torch.where(data >= zero_tol, f * data, torch.zeros_like(data)))
    elif aprox == "tv":
        assert (f <= rho).all(), "a should be less than rho for tv aprox"
        assert (f >= -rho).all(), "a should be greater than -rho for tv aprox"
        return -torch.sum(torch.where(data >= zero_tol, (torch.maximum(-f, -rho)) * data, torch.zeros_like(data)))
    else:
        raise NotImplementedError("Only kl, tv and balanced aprox implemented")


def _flat_grid_log_sinkhorn_reduction(f, d, ind, X, Y, epsilon):
    """
    if f and d are both Vi then ind= 0
    if f is Vi and d is Vj then ind=1
    """
    #something in numpy still

    # kernel computations - K @ a
    # main bottle neck
    if ind ==0:
        return log_reduction_ii(f, X, Y, epsilon, d) 
    elif ind == 1:
        return log_reduction_ij(f, X, Y, epsilon, d)


def Kd_dual_potential_reduction(dp, d, epsilon):
    """
    Kd reduction for debiasing potential update
    """
    # pick first edge becasue all edges should have the same barycentre node at edge[0]
    edge = list(dp.graph.edges)[0]

    # Symmetric reduction for debiasing term
    if "x1x1" in dp.data_dict[edge[0]] and "x2x2" in dp.data_dict[edge[0]]:
        s = _tensorised_sinkhorn_reduction(
            d,
            dp.data_dict[edge[0]]["x1x1"],
            dp.data_dict[edge[0]]["x2x2"],
            epsilon,
        )

    # Otherwise PyKeOps
    elif "grid" in dp.data_dict[edge[0]]:
        s = _flat_grid_sinkhorn_reduction(
            d,
            dp.data_dict[edge[0]]["grid"],
            dp.data_dict[edge[0]]["grid"],
            epsilon,
        )
    
    if torch.any(torch.isnan(s)) or torch.any(torch.isinf(s)):
        raise ValueError("Debiasing reduction NaN/inf detected", s.sum().item())
    
    if torch.any(s <= 0):
        raise ValueError("Debiasing reduction negative or zero values detected", s.min().item())
    
    return s

