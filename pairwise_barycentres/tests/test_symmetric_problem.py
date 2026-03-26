

import numpy as np
from scipy.spatial.distance import cdist
import torch
import pytest
from pwbarycentres import (
    asymmetric_sinkhorn_log_algorithm,
    generate_barycentredataprocessor,
    symmetric_algorithm,
    symmetric_cost
)
from pwbarycentres import asymmetric_matvec_cost as asymmetric_cost
import networkx as nx

torch.set_printoptions(precision=8)



@pytest.mark.parametrize(
    "n1, n2, members, L, grid_type",
        [ 
        (11, 10, 6, 0.9, "flat"),
        (11, 11, 7, 0.9, "flat"),
        (11, 12, 8, 0.9, "flat"),
        (8, 8, 4, 3.5, "tensor"),
        (9, 8, 3, 3.5, "tensor"),
        (8, 9, 2, 3.5, "tensor"),
        (12, 12, 3, 6.0, "tuple"),
        (12, 13, 3, 6.0, "tuple"),
        (12, 11, 3, 6.0, "tuple"),
    ],
)  # noqa: E501
def test_marginals_asym_log_bary_with_same_grid_uniform_density_without_debiasing(
    n1, n2, members, L, grid_type
):

    if grid_type == "flat":
        X = torch.cartesian_prod(
            torch.linspace(0, L, n1), torch.linspace(0, L, n2)
        ).type(torch.DoubleTensor)
    elif grid_type == "tensor":
        X = torch.stack(
            torch.meshgrid(
                torch.linspace(0, L, n1), torch.linspace(0, L, n2), indexing="ij"
            ),
            dim=-1,
        ).type(torch.DoubleTensor)
    elif grid_type == "tuple":
        X = (torch.linspace(0, L, n1), torch.linspace(0, L, n2))

    X_flat = torch.cartesian_prod(
            torch.linspace(0, L, n1), torch.linspace(0, L, n2)
        ).type(torch.DoubleTensor)
    
    data = []

    for m in range(members):
        data.append([None, None])  # uniform density, grid will equal everywhere

    # Just need to populate the class correctly:
    data_processor = generate_barycentredataprocessor(
        data, 
        barycentre_grid=X, 
        grid=X,
        potentials='f')

    # # run asymmetric sinkhorn algorithm
    # data_processor, barycentre, potential_error_list, barycentre_error_list, constraints_dict = (
    #     asymmetric_sinkhorn_log_algorithm(
    #         data_processor,
    #         epsilon=1 / np.sqrt(n1 * n2),
    #         rho=1.0,
    #         aprox="balanced",
    #         max_iterates=10,
    #         tol=1e-5,
    #         epsilon_annealing=False,
    #         debiasing=False,
    #         measure_constraints=True
    #     )
    # )

    epsilon = data_processor._torch_numpy_process(1 / np.sqrt(n1 * n2)).cpu()
        
    data = data_processor.data_dict[0]['density'].cpu().view(-1,1)
    f = data_processor.data_dict[0]['f'].cpu().clone().view(-1,1)
    leb = data_processor.data_dict[0]['cell_areas'].cpu()*torch.ones_like(f)
    
    K = torch.exp(-torch.cdist(X_flat, X_flat)**2 / epsilon/2)

    # 2 times last term cause L \otimes L . 
    f0 = epsilon*torch.log(data) - epsilon*torch.log(K@(torch.exp(f/epsilon)*leb))
    f = 0.5*(f + f0)
    
    sym_pot = symmetric_algorithm(data_processor, 0, epsilon, rho=1.0, aprox="balanced", max_iterates=1, tol=1e-9)
    sym_pot = sym_pot.cpu()

    assert torch.allclose(f.view(-1), sym_pot.view(-1), atol=1e-5), f"Symmetric potential not close to asymmetric potential, max diff {torch.abs(f-sym_pot).max()}"

    # Now run till convergence
    sym_pot = symmetric_algorithm(data_processor, 0, epsilon, rho=1.0, aprox="balanced", max_iterates=1000, tol=1e-12)
    sym_pot = sym_pot.cpu().view(-1, 1)

    # torch check of margainls?
    marg1 = (K * ((torch.exp(sym_pot/epsilon)).view(-1, 1) * (torch.exp(sym_pot/epsilon)*leb).view(1, -1))).sum(dim=1)
    marg1 = marg1.view(-1, 1)

    assert torch.allclose((marg1*leb).sum(), torch.tensor(1.0, dtype=torch.double), atol=1e-6), f"Marginal sum {(marg1*leb).sum()} not close to 1"
    assert torch.allclose((marg1*leb).view(-1), torch.ones_like(marg1.view(-1)) * (1/(n1*n2)), atol=1e-6), f"Marginal not close to uniform, max diff {torch.abs(marg1 - (1/(n1*n2))).max()}"

    # create torch evrsion of the cost
    torch_cost = 2*(sym_pot.view(-1)*leb.view(-1)*data.view(-1)).sum()
    
    temp = - epsilon * ((marg1*leb).sum() - torch.outer(leb.view(-1), leb.view(-1)).sum())
    torch_cost += temp

    cost = symmetric_cost(data_processor, 0, data_processor._torch_numpy_process(1 / np.sqrt(n1 * n2)), rho=1.0, aprox="balanced", max_iterates=1000, tol=1e-12)

    assert torch.allclose(cost, torch_cost, atol=1e-3)

if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
