# from asymmetric_sinkhorn_algorithm import 
from ..common.marginals import (
    calculate_node_marginal,
    _tensorised_marginal_reduction,
    _flat_grid_marginal_reduction,
    _calculate_debiasing_potential_symmetric_term
)
from graph_dp import SinkhornDataProcessor
from ..common.symmetric_problem import symmetric_cost
import torch
from ..common.utils import _dual_cost_data_term, _dual_cost_data_term_f_potential
from ..common.pykeops_formulas import c_pi_term
import numpy as np

def asymmetric_cost(
    dp: SinkhornDataProcessor,
    epsilon,
    rho,
    aprox: str,
    debiasing: bool = True,
    verbose: bool = False,
    fixed_barycentre=None,
    return_breakdown=False,
    ignore_const=False
):

    epsilon = dp._torch_numpy_process(epsilon).view(-1, 1)
    rho = dp._torch_numpy_process(rho)

    us_e = []
    uot_mu_mu = []
    if return_breakdown:
        cost_dict = dict()
    for edge in dp.graph.edges:
        weighting = dp.graph.edges[edge]["weight"]
        unbal_sinkhorn_div = _asymmetric_individual_edge_cost(
            dp, edge, epsilon, rho, aprox, debiasing, return_breakdown
        )
        if return_breakdown:
            cost, breakdown = unbal_sinkhorn_div
            us_e.append(cost.item() * weighting.item())
            _, primal_breakdown = _asymmetric_individual_edge_primal_cost(dp, edge, epsilon, rho, aprox, debiasing, return_breakdown=True, zero_tol=1e-12)
            cost_dict[edge] = {**breakdown, **primal_breakdown}
        else: 
            us_e.append(unbal_sinkhorn_div.item() * weighting.item())
        
        # solve UOT(mu_I, mu_I)
        if debiasing and not ignore_const:
            cost = symmetric_cost(dp, edge[1], epsilon, rho, aprox, max_iterates=2000, tol=1e-9)
            uot_mu_mu.append(cost.item() * weighting.item())
        else:
            uot_mu_mu.append(0)
        
        if return_breakdown:
            cost_dict[edge]["uot_mu_mu"] = uot_mu_mu[-1] 
            cost_dict[edge]["weight"] = weighting.item()
        
    if debiasing:
        # We need the last few terms
        if fixed_barycentre is None:
            d = dp.data_dict[edge[0]]["debiased_potential"]

            # calcualte <dkd - 1, L>
            debiasing_term = _calculate_debiasing_potential_symmetric_term(
                d, dp, edge[0], epsilon
            )

            debiasing_term1 = debiasing_term.item()*-0.5*epsilon.item()

            # add -elogd,xi
            debiasing_term2 = -epsilon.item()*torch.sum(torch.where(d > 0, dp.data_dict[edge[0]]["density"] * torch.log(d), torch.zeros_like(d)))

            debiasing_term = debiasing_term1 + debiasing_term2
            
            if return_breakdown:
                cost_dict["debiasing_term1"] = debiasing_term1
                cost_dict["debiasing_term2"] = debiasing_term2.item()

        else:
            # calcualte UOT_(e,e)
            # can choose any edge since they should be the same
            assert np.allclose(fixed_barycentre, dp.data_dict[edge[0]]["density"].cpu().numpy()), "Fixed barycentre should be the same as the barycentre node in the edge"
            debiasing_term = symmetric_cost(dp, edge[0], epsilon, rho, aprox='balanced', max_iterates=2000, tol=1e-9)
            debiasing_term *= -0.5
            print("Debiasing term calculated using fixed barycentre: ", debiasing_term.item())
            if return_breakdown:
                cost_dict["debiasing_term"] = debiasing_term.item()

        full_cost = sum(us_e) + debiasing_term.item() - np.stack(uot_mu_mu).sum()/2
        full_cost = full_cost.item()
    else:
        full_cost = sum(us_e)
    
    if return_breakdown:
        cost_dict= dict(
            total_cost=full_cost,
            unbalanced_sinkhorn_terms=us_e,
            uot_mu_mu_terms=uot_mu_mu if debiasing else None,
            debiasing_term=debiasing_term.item() if debiasing else None,
            epsilon=epsilon.item(),
            rho=rho.item(),
            aprox=aprox,
            debiasing=debiasing,
            subbreakdown=cost_dict
        )
        return full_cost, us_e, cost_dict
    else:
        return full_cost, us_e
    

def _asymmetric_individual_edge_cost(dp, edge, epsilon, rho, aprox, debiasing, return_breakdown=False):
    bary_node = edge[0]
    data_node = edge[1]
    
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

    # <e^{f+g-c}>
    term3 = calculate_node_marginal(dp, bary_node, epsilon, debiasing=False)[0].sum()

    # final constant <Kd> 
    term4 = 1  # ~ sum_n 1/n

    cost = term1 + term2 - epsilon * (term3 - term4)
    if return_breakdown:
        return cost, dict(term1=term1, term2=term2, term3=term3, term4=term4)
    return cost


def _asymmetric_individual_edge_primal_cost(dp, edge, epsilon, rho, aprox, debiasing, return_breakdown=False, zero_tol=1e-12):
    bary_node = edge[0]
    data_node = edge[1]
    
    f = dp.data_dict[data_node]["f"]
    g = dp.data_dict[bary_node]["f"]

    # transport term
    c_pi = c_pi_term(f, g, dp.data_dict[data_node]["grid"], dp.data_dict[bary_node]["grid"], epsilon)
    
    # divergence term: forced debaising false because we don't attach d anymore to the kernel
    bary_marginal = calculate_node_marginal(dp, bary_node, epsilon, debiasing=False)[0]
    data_marginal = calculate_node_marginal(dp, data_node, epsilon, debiasing=False)[0]

    assert torch.allclose(bary_marginal, dp.data_dict[bary_node]["density"]), "Marginals should have the same total mass"
    # This term has to be eaul otherwise the cost is infinite, so we can ignore it in the cost calculation

    if aprox == "balanced":
        assert torch.allclose(data_marginal, dp.data_dict[data_node]["density"]), "Marginals should have the same total mass"
        divergence_term = 0
    elif aprox == "kl":
        temp = torch.where(dp.data_dict[data_node]["density"] > zero_tol, torch.log(data_marginal / dp.data_dict[data_node]["density"]), torch.zeros_like(data_marginal))
        divergence_term = torch.sum(data_marginal * (temp - 1)) + torch.sum(dp.data_dict[data_node]["density"])
    elif aprox == "tv":
        # same as  |A/B - 1| * B and more stable
        divergence_term = torch.sum(torch.abs(data_marginal - dp.data_dict[data_node]["density"]))
    else:
        raise ValueError("Unknown aprox type")

    cost = c_pi - rho * divergence_term
    if return_breakdown:
        return cost, dict(c_pi=c_pi, divergence_term=divergence_term)
    return cost


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

