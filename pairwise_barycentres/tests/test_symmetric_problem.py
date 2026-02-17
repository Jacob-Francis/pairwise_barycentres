

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
        
    K = torch.exp(-torch.cdist(X_flat, X_flat)**2 / epsilon/2) #/ np.prod(X_flat.shape[0])**2
    data = data_processor.data_dict[0]['density'].cpu().view(-1,1)
    f = data_processor.data_dict[0]['f'].cpu().clone().view(-1,1)
    # 2 times last term cause L \otimes L . 
    f0 = epsilon*torch.log(data) - epsilon*torch.log(K@torch.exp(f/epsilon)) -2* epsilon* np.log(1/np.prod(data.shape))
    f = 0.5*(f + f0)
    
    sym_pot = symmetric_algorithm(data_processor, 0, epsilon, rho=1.0, aprox="balanced", max_iterates=1, tol=1e-9)
    sym_pot = sym_pot.cpu()

    print(f.shape, sym_pot.shape)
    assert torch.allclose(f.view(-1), sym_pot.view(-1), atol=1e-5), f"Symmetric potential not close to asymmetric potential, max diff {torch.abs(f-sym_pot).max()}"

    # Now run till convergence
    sym_pot = symmetric_algorithm(data_processor, 0, epsilon, rho=1.0, aprox="balanced", max_iterates=1000, tol=1e-12)
    sym_pot = sym_pot.cpu()

    # torch check of margainls?
    marg1 = (K * (torch.exp(sym_pot/epsilon).view(-1, 1) * torch.exp(sym_pot/epsilon).view(1, -1))).sum(dim=1) / np.prod(data.shape)**2

    assert torch.allclose(marg1.sum(), torch.tensor(1.0, dtype=torch.double), atol=1e-6), f"Marginal sum {marg1.sum()} not close to 1"
    assert torch.allclose(marg1.view(-1), torch.ones_like(marg1.view(-1)) * (1/(n1*n2)), atol=1e-6), f"Marginal not close to uniform, max diff {torch.abs(marg1 - (1/(n1*n2))).max()}"

    # create torch evrsion of the cost
    torch_cost = 2*sym_pot.sum()*1/(n1*n2)
    
    temp = - epsilon * ((K * (torch.exp(sym_pot/epsilon).view(-1, 1) * torch.exp(sym_pot/epsilon).view(1, -1))).sum() * (1/(n1*n2))**2 - 1)
    torch_cost += temp

    cost = symmetric_cost(data_processor, 0, data_processor._torch_numpy_process(1 / np.sqrt(n1 * n2)), rho=1.0, aprox="balanced", max_iterates=1000, tol=1e-12)
    print(cost, torch_cost)
    assert torch.allclose(cost, torch_cost, atol=1e-3)

