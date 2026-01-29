# from asymmetric_sinkhorn_algorithm import 
from .marginals import (
    calculate_node_marginal,
    _tensorised_marginal_reduction,
    _flat_grid_marginal_reduction,
)
from graph_dp import SinkhornDataProcessor
from .symmetric_problem import symmetric_cost
import torch
from .utils import _dual_cost_data_term, _dual_cost_data_term_f_potential

def asymmetric_cost(
    dp: SinkhornDataProcessor,
    epsilon,
    rho,
    aprox: str,
    debiasing: bool = True,
    verbose: bool = False,
    fixed_barycentre=None,
    return_breakdown=False
):

    epsilon = dp._torch_numpy_process(epsilon).view(-1, 1)
    rho = dp._torch_numpy_process(rho)

    us_e = []
    uot_mu_mu = []
    for edge in dp.graph.edges:
        weighting = dp.graph.edges[edge]["weight"]
        unbal_sinkhorn_div = _asymmetric_individual_edge_cost(
            dp, edge, epsilon, rho, aprox, debiasing
        )
        us_e.append(unbal_sinkhorn_div * weighting)
        
        # solve UOT(mu_I, mu_I)
        if debiasing:
            cost = symmetric_cost(dp, edge[1], edge, epsilon, rho, aprox, max_iterates=2000, tol=1e-9)
            uot_mu_mu.append(cost * weighting)

    if debiasing:
        # We need the last few terms
        if fixed_barycentre is None:
            d = dp.data_dict[edge[0]]["debiased_potential"]
            # calcualte <d-1 K (d-1)>
            debiasing_term = _calculate_debiasing_potential_symmetric_term(
                d, dp, edge[0], epsilon
            )
        else:
            # calcualte UOT_(e,e)
            # can choose any edge since they should be the same
            debiasing_term = symmetric_cost(dp, edge[0], edge, epsilon, rho, aprox='balanced', max_iterates=2000, tol=1e-9)

        full_cost = sum(us_e) - epsilon * debiasing_term / 2 - epsilon* torch.stack(uot_mu_mu).sum()/2
    else:
        full_cost = sum(us_e)
    
    if return_breakdown:
        cost_dict= dict(
            total_cost=full_cost.item(),
            unbalanced_sinkhorn_terms=us_e,
            uot_mu_mu_terms=uot_mu_mu if debiasing else None,
            debiasing_term=debiasing_term if debiasing else None,
            epsilon=epsilon.item(),
            rho=rho.item(),
            aprox=aprox,
            debiasing=debiasing
        )
        return full_cost, us_e, cost_dict
    else:
        return full_cost, us_e
    

def _asymmetric_individual_edge_cost(dp, edge, epsilon, rho, aprox, debiasing):
    bary_node = edge[0]
    data_node = edge[1]


    try: 
        a = dp.data_dict[data_node]["a"]
        b = dp.data_dict[bary_node]["a"]

        # Have sufficent information for term 1 and term 2 of dual cost

        # -<-f, alpha>
        term1 = _dual_cost_data_term(
            a, dp.data_dict[data_node]["density"], aprox, epsilon, rho
        )
        # -<-g, beta>
        term2 = _dual_cost_data_term(
            b, dp.data_dict[bary_node]["density"], "balanced", epsilon, rho
        )
    except KeyError:
        f = dp.data_dict[data_node]["f"]
        g = dp.data_dict[bary_node]["f"]

        # -<-f, alpha>
        term1 = _dual_cost_data_term_f_potential(
            f, dp.data_dict[data_node]["density"], aprox, epsilon, rho
        )
        # -<-g, beta>
        term2 = _dual_cost_data_term_f_potential(
            g, dp.data_dict[bary_node]["density"], "balanced", epsilon, rho
        )
    
    # <e^{f+g-c}d>=<a.Kbd>
    term3 = calculate_node_marginal(dp, bary_node, epsilon, debiasing)[0].sum()

    # final constant <Kd> 
    term4 = _calculate_dual_cost_constant(dp, edge, epsilon, debiasing)

    return term1 + term2 - epsilon * (term3 - term4)


def _calculate_dual_cost_constant(dp, edge, epsilon, debiasing):
    """
    we can hack the marginal reductions for find the cost constant summation <K>
    by using ones vectors for ai and bj
    """

    bary_node = edge[0]
    data_node = edge[1]

    # only need these for the shape so doesn't matter which one we use
    if 'a' in dp.data_dict[data_node] and 'a' in dp.data_dict[bary_node]:
        string = 'a'
    elif 'f' in dp.data_dict[data_node] and 'f' in dp.data_dict[bary_node]:
        string = 'f'
    else:
        raise Warning("No potentials or scaling factors attached to either node")

    if debiasing:
        if "debiased_potential" in dp.data_dict[bary_node]:
            b = (
                torch.ones_like(dp.data_dict[bary_node][string])
                * dp.data_dict[bary_node]["debiased_potential"]
            )
            a = torch.ones_like(dp.data_dict[data_node][string])
        elif "debiased_potential" in dp.data_dict[data_node]:
            raise Warning("No debiasing potentials should be attached to the data")
        else:
            raise Warning(
                "No debiasing potentials attached to either node, yet using debiasing"
            )

        if (
            "debiased_potential" in dp.data_dict[bary_node]
            and "debiased_potential" in dp.data_dict[data_node]
        ):
            raise Warning(
                "Both nodes have debiasing potentials attached, this is unexpected behaviour"
            )
    else:
        a = torch.ones_like(dp.data_dict[data_node][string])
        b = torch.ones_like(dp.data_dict[bary_node][string])

    # These terms are the same as the marginal term reductions
    if "x1y1" in dp.data_dict[edge] and "x2y2" in dp.data_dict[edge]:
        # we can tensorise
        cost_constant = _tensorised_marginal_reduction(
            dp.data_dict[edge]["x1y1"],  # either order tensorise_f will sort it
            dp.data_dict[edge]["x2y2"],
            epsilon,
            a,
            b,
        )
    elif "grid" in dp.data_dict[data_node] and "grid" in dp.data_dict[bary_node]:
        # we can use PyKeOps
        cost_constant = _flat_grid_marginal_reduction(
            dp.data_dict[data_node]["grid"],
            dp.data_dict[bary_node]["grid"],
            epsilon,
            a,
            b,
        )

    return cost_constant.sum()

def _calculate_debiasing_potential_symmetric_term(d, dp, node, epsilon):
    """
    we can hack the marginal reductions for find the cost constant summation <K>
    by using ones vectors for ai and bj
    """

    if "x1x1" in dp.data_dict[node] and "x2x2" in dp.data_dict[node]:
        # we can tensorise
        cost_constant = _tensorised_marginal_reduction(
            dp.data_dict[node]["x1x1"],  # either order tensorise_f will sort it
            dp.data_dict[node]["x2x2"],
            epsilon,
            d - 1, # look here!
            d - 1,
        )
    elif "grid" in dp.data_dict[node]:
        # we can use PyKeOps
        print('devices:', dp.data_dict[node]["grid"].device, d.device, epsilon.device)

        cost_constant = _flat_grid_marginal_reduction(
            dp.data_dict[node]["grid"],
            dp.data_dict[node]["grid"],
            epsilon,
            d.view(-1,1) - 1,
            d.view(-1,1) - 1,
        )

    return cost_constant.sum()

