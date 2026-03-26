import torch
import numpy as np
from graph_dp import SinkhornDataProcessor
from .pykeops_formulas import chizat_marginals, fg_reduction_ii, fg_reduction_ij
from .utils import tensorise_f


def ot_marginals(dp: SinkhornDataProcessor, epsilon, debiasing=True, nodes=None):
    """
    Calculate the marginals at the node(s) specified

    ToDo: Its not ideal that we have have to pass on debiasing... hummm
    """

    epsilon = dp._torch_numpy_process(epsilon).view(-1, 1)

    # Process nodes input:
    if nodes is None:
        nodes = list(dp.graph.nodes)
    elif isinstance(nodes, int):
        nodes = [nodes]
    elif isinstance(nodes, list):
        pass
    else:
        raise TypeError("Invalid nodes input")

    # Compute marginals
    marginals = {}
    for node in nodes:
        marginals[node] = {}
        marginals[node]["marginal"], marginals[node]["error"] = calculate_node_marginal(
            dp, node, epsilon, debiasing
        )

    return marginals


def calculate_node_marginal(dp: SinkhornDataProcessor, node, epsilon, debiasing):
    """
    Calculate the marginal for a specific node.
    """
    # Get the node's data
    node_data = dp.data_dict[node]

    # I have to look across my neighbours to see whos connected, and sum inwards;
    # I'm going to first do this only for the pairwise approach this its a lot more simple than
    # the general graph case

    # Compute the marginal - depending on tensorisation or not
    for neighbour in dp.graph.neighbors(node):
        edge = (
            (node, neighbour)
            if (node, neighbour) in list(dp.graph.edges)
            else (neighbour, node)
        )

        if 'f' in dp.data_dict[node] and 'f' in dp.data_dict[neighbour]:
            marginal = fg_potential_marginal_reduction(dp, node, epsilon, neighbour, edge)
        elif 'a' in dp.data_dict[node] and 'a' in dp.data_dict[neighbour]:
            marginal = ab_potential_marginal_reduction(dp, node, epsilon, debiasing, neighbour, edge)
          # we never acttch the potential anymore
        else:
            raise Warning(
                "Node potentials not found or do not match between nodes"
            )

    # mat vec version and log are inconsistent ; going with log
    error = torch.norm(marginal - node_data["density"], p=float("inf")).item()

    # print(f"marginal error for node {node}: {error}, sum of marginal: {marginal.sum().item()}, sum of density*cell_areas: {(node_data['density']*node_data['cell_areas']).sum().item()}")
    return marginal, error

def ab_potential_marginal_reduction(dp, node, epsilon, debiasing, neighbour, edge):
    if debiasing:
        if "debiased_potential" in dp.data_dict[node]:
            b = dp.data_dict[node]["a"] * dp.data_dict[node]["debiased_potential"]
            a = dp.data_dict[neighbour]["a"]
        elif "debiased_potential" in dp.data_dict[neighbour]:
            b = dp.data_dict[node]["a"]
            a = (
                    dp.data_dict[neighbour]["a"]
                    * dp.data_dict[neighbour]["debiased_potential"]
                )
        else:
            raise Warning(
                    "No debiasing potentials attached to either node, yet using debiasing"
                )

        if (
                "debiased_potential" in dp.data_dict[node]
                and "debiased_potential" in dp.data_dict[neighbour]
            ):
            raise Warning(
                    "Both nodes have debiasing potentials attached, this is unexpected behaviour"
                )
    else:
        a = dp.data_dict[neighbour]["a"]
        b = dp.data_dict[node]["a"]

    if "x1y1" in dp.data_dict[edge] and "x2y2" in dp.data_dict[edge]:
            # we can tensorise
        marginal = _tensorised_marginal_reduction(
                dp.data_dict[edge]["x1y1"],  # either order tensorise_f will sort it
                dp.data_dict[edge]["x2y2"],
                epsilon,
                a,
                b,
            )
    elif "grid" in dp.data_dict[node] and "grid" in dp.data_dict[neighbour]:
            # we can use PyKeOps
        marginal = _flat_grid_marginal_reduction(
                dp.data_dict[neighbour]["grid"],
                dp.data_dict[node]["grid"],
                epsilon,
                a,
                b,
            )
        
    return marginal


def fg_potential_marginal_reduction(dp, node, epsilon, neighbour, edge):

    f = dp.data_dict[neighbour]["f"]
    g = dp.data_dict[node]["f"]

    assert torch.is_tensor(dp.data_dict[neighbour]['cell_areas'])
    if dp.data_dict[neighbour]['cell_areas'].ndim == 0:
        leb_f = dp.data_dict[neighbour]['cell_areas']*torch.ones_like(dp.data_dict[neighbour]["f"])
    else:
        leb_f = dp.data_dict[neighbour]['cell_areas']

    assert torch.is_tensor(dp.data_dict[node]['cell_areas'])
    if dp.data_dict[node]['cell_areas'].ndim == 0:
        leb_g = dp.data_dict[node]['cell_areas']*torch.ones_like(dp.data_dict[node]["f"])
    else:
        leb_g = dp.data_dict[node]['cell_areas']


    if leb_f.ndim >0 :
        assert f.shape == leb_f.shape
    if leb_g.ndim >0 :
        assert g.shape == leb_g.shape

    if "x1y1" in dp.data_dict[edge] and "x2y2" in dp.data_dict[edge]:
    
        marginal = _tensorised_marginal_reduction(
                dp.data_dict[edge]["x1y1"],  # either order tensorise_f will sort it
                dp.data_dict[edge]["x2y2"],
                epsilon,
                torch.exp(f/epsilon)*leb_f,
                torch.exp(g/epsilon),
            )
    elif "grid" in dp.data_dict[node] and "grid" in dp.data_dict[neighbour]:
        # we can use PyKeOps
        marginal = fg_reduction_ii(
            Fi=f,
            Gj=g,
            Xi=dp.data_dict[neighbour]["grid"],
            Yj=dp.data_dict[node]["grid"],
            epsilon=epsilon.view(-1,1),
            ai=leb_f,
        )
        marginal = marginal

    assert marginal.shape == leb_g.shape

    return marginal


# If debiasing we can 'attach' the debiasing potential to the marginal reduction
# of a or b.
def _tensorised_marginal_reduction(x1y1, x2y2, epsilon, ai, bj):
    return (
        tensorise_f(
            torch.exp(-x1y1 / epsilon),
            torch.exp(-x2y2 / epsilon),
            ai,
        )
        * bj
    )


def _flat_grid_marginal_reduction(X, Y, epsilon, ai, bj):
    return chizat_marginals(
        X,
        Y,
        epsilon,
        ai,
        bj,
    )



def _calculate_debiasing_potential_symmetric_term(d, dp, node, epsilon, cell_areas, leb=True):
    """
    we can hack the marginal reductions for find the cost constant summation <K>
    by using ones vectors for ai and bj
    """

    if cell_areas.ndim >0:
        assert cell_areas.shape == d.shape
    

    if "x1x1" in dp.data_dict[node] and "x2x2" in dp.data_dict[node]:
        # we can tensorise
        cost_constant = _tensorised_marginal_reduction(
            dp.data_dict[node]["x1x1"],  # either order tensorise_f will sort it
            dp.data_dict[node]["x2x2"],
            epsilon,
            d*cell_areas if leb else (d - 1)*cell_areas, # look here!
            d*cell_areas if leb else (d - 1)*cell_areas,
        )
    elif "grid" in dp.data_dict[node]:
        # we can use PyKeOps
        cost_constant = _flat_grid_marginal_reduction(
            dp.data_dict[node]["grid"],
            dp.data_dict[node]["grid"],
            epsilon,
            d.view(-1,1)*cell_areas if leb else (d.view(-1,1) - 1)*cell_areas,
            d.view(-1,1)*cell_areas if leb else (d.view(-1,1) - 1)*cell_areas,
        )

    if leb:
        return cost_constant.sum() 
    else:
        return cost_constant.sum()