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
from ..common.pykeops_formulas import c_pi_term, log_pi_term
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
    ignore_const=False,
    primal_cost=False,
    zero_tol=1e-40
):

    epsilon = dp._torch_numpy_process(epsilon).view(-1, 1)
    rho = dp._torch_numpy_process(rho)

    us_e = []
    uot_mu_mu = []
    if primal_cost:
        primal_costs = []
    if return_breakdown:
        cost_dict = dict()
    for edge in dp.graph.edges:
        weighting = dp.graph.edges[edge]["weight"]
        unbal_sinkhorn_div = _asymmetric_individual_edge_cost(
            dp, edge, epsilon, rho, aprox, return_breakdown, zero_tol=zero_tol
        )

        if return_breakdown and primal_cost:
            cost, breakdown = unbal_sinkhorn_div
            us_e.append(cost.item() * weighting.item())
            primal_c, primal_breakdown = _asymmetric_individual_edge_primal_cost(dp, edge, epsilon, rho, aprox, return_breakdown=True, zero_tol=zero_tol)
            cost_dict[edge] = {**breakdown, **primal_breakdown}
            primal_costs.append(primal_c.item() * weighting.item())
        elif return_breakdown:
            cost, breakdown = unbal_sinkhorn_div
            us_e.append(cost.item() * weighting.item())
            cost_dict[edge] = {**breakdown}
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
            cell_areas = dp.data_dict[edge[0]]["cell_areas"]

            if cell_areas.ndim > 0:
                assert cell_areas.shape == d.shape, "Cell areas and debiasing potential should have the same shape"

            # calcualte <dkd, L>
            debiasing_term = _calculate_debiasing_potential_symmetric_term(
                d, dp, edge[0], epsilon, cell_areas,
            )

            # final constant <1, L> 
            if cell_areas.ndim == 0:
                term4 = (np.prod(d.shape)*cell_areas.item())**2
            elif cell_areas.ndim > 0 :
                term4 = cell_areas.sum()**2
            else:
                raise ValueError("Cell areas should either both be scalars or both be tensors of the same shape")
            debiasing_term -= term4
            debiasing_term1 = debiasing_term.item()*0.5*epsilon.item()

            # add -elogd,xi
            dtemp = torch.clamp(d, min=1e-40) # to avoid log(0)
            assert dp.data_dict[edge[0]]["density"].shape == dtemp.shape, "Density and debiasing potential should have the same shape"
            debiasing_term2 = -epsilon.item()*(dp.data_dict[edge[0]]["density"] * torch.log(dtemp) * cell_areas).sum().item()

            debiasing_term = debiasing_term1 + debiasing_term2
            
            if return_breakdown:
                cost_dict["debiasing_term1"] = debiasing_term1
                cost_dict["debiasing_term2"] = debiasing_term2

        else:
            # calcualte UOT_(e,e)
            # can choose any edge since they should be the same
            assert np.allclose(fixed_barycentre, dp.data_dict[edge[0]]["density"].cpu().numpy()), "Fixed barycentre should be the same as the barycentre node in the edge"
            debiasing_term = symmetric_cost(dp, edge[0], epsilon, rho, aprox='balanced', max_iterates=2000, tol=1e-9)
            debiasing_term *= -0.5
            debiasing_term = debiasing_term.item()
            print("Debiasing term calculated using fixed barycentre: ", debiasing_term)
            if return_breakdown:
                cost_dict["debiasing_term"] = debiasing_term

        full_cost = sum(us_e) + debiasing_term - np.stack(uot_mu_mu).sum()/2
        full_cost = full_cost
    else:
        full_cost = sum(us_e)

    if primal_cost:
        print('primal cost', sum(primal_costs))
    
    if return_breakdown:
        cost_dict= dict(
            total_cost=full_cost,
            unbalanced_sinkhorn_terms=us_e,
            uot_mu_mu_terms=uot_mu_mu if debiasing else None,
            debiasing_term=debiasing_term if debiasing else None,
            epsilon=epsilon.item(),
            rho=rho.item(),
            aprox=aprox,
            debiasing=debiasing,
            subbreakdown=cost_dict
        )
        return full_cost, us_e, cost_dict
    else:
        return full_cost, us_e
    

def _asymmetric_individual_edge_cost(dp, edge, epsilon, rho, aprox, return_breakdown=False, zero_tol=1e-40):
    bary_node = edge[0]
    data_node = edge[1]
    
    f = dp.data_dict[data_node]["f"]
    g = dp.data_dict[bary_node]["f"]

    cell_areas_g = dp.data_dict[bary_node]["cell_areas"]
    cell_areas_f = dp.data_dict[data_node]["cell_areas"]

    # -<-f, alpha>
    term1 = _dual_cost_data_term_f_potential(
        f, dp.data_dict[data_node]["density"], aprox, epsilon, rho, cell_areas_f, zero_tol=zero_tol
    )
    # -<-g, beta>
    term2 = _dual_cost_data_term_f_potential(
        g, dp.data_dict[bary_node]["density"], "balanced", epsilon, rho, cell_areas_g, zero_tol=zero_tol
    )

    # <e^{f+g-c}, L>
    term3 = calculate_node_marginal(dp, bary_node, epsilon, debiasing=False)[0]
    term3 *= cell_areas_g
    term3 = term3.sum()

    # final constant <1, L> 
    if cell_areas_f.ndim == 0 and cell_areas_g.ndim == 0:
        term4 = np.prod(f.shape)*cell_areas_f.item()* np.prod(g.shape)*cell_areas_g.item()
    elif cell_areas_f.ndim > 0 and cell_areas_g.ndim > 0:
        term4 = cell_areas_f.sum() * cell_areas_g.sum()
    else:
        raise ValueError("Cell areas should either both be scalars or both be tensors of the same shape")

    cost = term1 + term2 - epsilon * (term3 - term4)

    if return_breakdown:
        return cost, dict(dual_term1=term1, dual_term2=term2, dual_term3=term3, dual_term4=term4)
    return cost


def _asymmetric_individual_edge_primal_cost(dp, edge, epsilon, rho, aprox, return_breakdown=False, zero_tol=1e-40):
    bary_node = edge[0]
    data_node = edge[1]
    
    f = dp.data_dict[data_node]["f"]
    g = dp.data_dict[bary_node]["f"]
    # cell areas
    cell_areas_g = dp.data_dict[bary_node]["cell_areas"]
    if cell_areas_g.ndim == 0:
        cell_areas_g = torch.ones_like(g)*cell_areas_g
    cell_areas_f = dp.data_dict[data_node]["cell_areas"]
    if cell_areas_f.ndim == 0:
        cell_areas_f = torch.ones_like(f)*cell_areas_f

    # transport term
    # tuple or not
    if isinstance(dp.data_dict[data_node]["grid"], tuple):
        grid_data = torch.cartesian_prod(*dp.data_dict[data_node]["grid"])
        grid_bary = torch.cartesian_prod(*dp.data_dict[bary_node]["grid"])
    else:
        grid_data = dp.data_dict[data_node]["grid"]
        grid_bary = dp.data_dict[bary_node]["grid"]
    
    c_pi = c_pi_term(f, g, dp._torch_numpy_process(grid_data), dp._torch_numpy_process(grid_bary), epsilon, cell_areas_f, cell_areas_g)
    
    # divergence term: forced debaising false because we don't attach d anymore to the kernel
    bary_marginal = calculate_node_marginal(dp, bary_node, epsilon, debiasing=False)[0]
    data_marginal = calculate_node_marginal(dp, data_node, epsilon, debiasing=False)[0]

    assert torch.allclose(bary_marginal.sum(), data_marginal.sum()), "Marginals should have the same total mass, but got {} and {}".format(bary_marginal.sum().item(), data_marginal.sum().item())

    assert torch.allclose(bary_marginal, dp.data_dict[bary_node]["density"]), "Marginals should have the same total mass"
    # This term has to be eqaual otherwise the cost is infinite, so we can ignore it in the cost calculation

    if aprox == "balanced": # it depend which update was last if it can exactly match or not, so we use a tolerance
        assert torch.allclose(data_marginal*cell_areas_f, dp.data_dict[data_node]["density"]*cell_areas_f, rtol=1e-10, atol=1e-4), "Marginals should have the same total mass {} vs {}, norm {}".format(data_marginal.sum().item(), dp.data_dict[data_node]["density"].sum().item(), (data_marginal*cell_areas_f - dp.data_dict[data_node]["density"]*cell_areas_f).norm().item())
        divergence_term = 0
    elif aprox == "kl":
        data = torch.clamp(dp.data_dict[data_node]["density"], min=zero_tol)
        # print('data marginal sum check', data_marginal.sum().item(), data.sum().item())
        divergence_term = torch.sum(data_marginal * (torch.log(data_marginal/ data) - 1) * cell_areas_f) + torch.sum(dp.data_dict[data_node]["density"] * cell_areas_f)
    elif aprox == "tv":
        # same as  |A/B - 1| * B and more stable
        divergence_term = torch.sum(torch.abs(data_marginal - dp.data_dict[data_node]["density"])* cell_areas_f)
    else:
        raise ValueError("Unknown aprox type")

    # entropy term: epsilon*((torch.log(pi/leb) - 1)*pi) + sum(leb)
    sum_pi = (bary_marginal*cell_areas_g).sum()
    sum_leb = cell_areas_f.sum()*cell_areas_g.sum()
    log_pi = log_pi_term(f, g, dp._torch_numpy_process(grid_data), dp._torch_numpy_process(grid_bary), epsilon, cell_areas_f, cell_areas_g)
    entropy = epsilon*(log_pi - sum_pi + sum_leb)

    cost = c_pi + rho * divergence_term + entropy

    # print(f"Cost: {cost}, c_pi: {c_pi}, divergence_term: {rho * divergence_term}, entropy: {entropy}")
    
    if return_breakdown:
        return cost, dict(primal_c_pi=c_pi, primal_divergence_term=rho * divergence_term, primal_entropy=entropy)
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

