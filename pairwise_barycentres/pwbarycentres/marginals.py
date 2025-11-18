import torch
import numpy as np
from .data_processing import SinkhornDataProcessor
from .pykeops_formulas import chizat_marginals
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
        marginals[node]['marginal'], marginals[node]['error'] = calculate_node_marginal(dp, node, epsilon, debiasing)
    
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
        edge = (node, neighbour) if (node, neighbour) in list(dp.graph.edges) else (neighbour, node)

        if debiasing:
            if 'debiased_potential' in dp.data_dict[node]:
                b = dp.data_dict[node]['a'] * dp.data_dict[node]['debiased_potential']
                a = dp.data_dict[neighbour]['a']
            elif 'debiased_potential' in dp.data_dict[neighbour]:
                b = dp.data_dict[node]['a']
                a = dp.data_dict[neighbour]['a'] * dp.data_dict[neighbour]['debiased_potential']
            else:
                raise Warning("No debiasing potentials attached to either node, yet using debiasing")

            if 'debiased_potential' in dp.data_dict[node] and 'debiased_potential' in dp.data_dict[neighbour]:
                raise Warning("Both nodes have debiasing potentials attached, this is unexpected behaviour")
        else:
            a = dp.data_dict[neighbour]['a']
            b = dp.data_dict[node]['a']  

        if 'x1y1' in dp.data_dict[edge] and 'x2y2' in dp.data_dict[edge]:
            # we can tensorise 
            marginal = _tensorised_marginal_reduction(
                dp.data_dict[edge]['x1y1'], # either order tensorise_f will sort it
                dp.data_dict[edge]['x2y2'],
                epsilon,
                a,
                b,
            )
        elif 'grid' in dp.data_dict[node] and 'grid' in dp.data_dict[neighbour]:
            # we can use PyKeOps
            marginal = _flat_grid_marginal_reduction(
                dp.data_dict[neighbour]['grid'],
                dp.data_dict[node]['grid'],
                epsilon,
                a,
                b,
            )

    error = torch.norm(
        marginal - node_data['density'],
        p=float('inf')
    ).item()

    return marginal, error


# If debiasing we can 'attach' the debiasing potential to the marginal reduction
# of a or b.
def _tensorised_marginal_reduction(x1y1, x2y2, epsilon, ai, bj):
    return tensorise_f(
        torch.exp(-x1y1/epsilon),
        torch.exp(-x2y2/epsilon),
        ai,
    ) * bj

def _flat_grid_marginal_reduction(X, Y, epsilon, ai, bj):
    return chizat_marginals(
        X,
        Y,
        epsilon,
        ai,
        bj,
    )
