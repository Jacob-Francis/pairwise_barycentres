import numpy as np
from scipy.spatial.distance import cdist
import torch
import pytest
from pwbarycentres import (
    asymmetric_sinkhorn_algorithm,
    generate_barycentredataprocessor,
    asymmetric_cost,
    ot_marginals,
    _calculate_debiasing_potential_symmetric_term,
    process_dict_for_barycentre

)
from pwbarycentres import asymmetric_cost as asym_cost
import networkx as nx

torch.set_printoptions(precision=8)




@pytest.mark.parametrize(
    "n1, n2, members, L, grid_type",
    [
        (8, 8, 8, 3.5, "tensor"),
        (9, 8, 5, 3.5, "tensor"),
        (8, 9, 3, 3.5, "tensor"),
        (11, 11, 3, 0.9, "flat"),
        (11, 10, 2, 0.9, "flat"),
        (11, 12, 3, 0.9, "flat"),
        (12, 12, 3, 6.0, "tuple"),
        (12, 13, 3, 6.0, "tuple"),
        (12, 11, 3, 6.0, "tuple"),
    ],
)  # noqa: E501
def test_marginals_asym_bary_with_same_grid_uniform_density_without_debiasing(
    n1, n2, members, L, grid_type
):

    dx = L / n1
    dy = L / n2
    if grid_type == "flat":
        X = torch.cartesian_prod(
            torch.linspace(dx/2, L - dx/2, n1), torch.linspace(dy/2, L - dy/2, n2)
        ).type(torch.DoubleTensor)
    elif grid_type == "tensor":
        X = torch.stack(
            torch.meshgrid(
                torch.linspace(dx/2, L - dx/2, n1), torch.linspace(dy/2, L - dy/2, n2), indexing="ij"
            ),
            dim=-1,
        ).type(torch.DoubleTensor)
    elif grid_type == "tuple":
        X = (torch.linspace(dx/2, L - dx/2, n1), torch.linspace(dy/2, L - dy/2, n2))

    data = []

    for m in range(members):
        data.append([None, None])  # uniform density, grid will equal everywhere

    # generate the barycentre dataprocessor class which will store all objects
    # of interest. It will also create the correct graph, and given no density of graphs
    # will create uniform densities on the grids
    dp = generate_barycentredataprocessor(data, barycentre_grid=X, grid=X)
    cell_areas = dp.data_dict[0]['cell_areas']
    process_dict_for_barycentre(dp, debiasing=True)

    # Don't actually need to solve anything just attach a debiasing potential
    d = dp._torch_numpy_process(torch.abs(torch.randn_like(dp.data_dict[0]["density"])))
    epsilon = dp._torch_numpy_process(1 / np.sqrt(n1 * n2)).view(-1,1)


    # something weird is happening with the grids
    dp, barycentre, potential_error_list, barycentre_error_list = (
        asymmetric_sinkhorn_algorithm(
            dp,
            epsilon=epsilon,
            rho=1.0,
            aprox="balanced",
            max_iterates=0,
            tol=1e-7,
            epsilon_annealing=False,
            debiasing=False,
        )
    )

    dkd = _calculate_debiasing_potential_symmetric_term(d, dp, 0, epsilon, cell_areas, leb=False)

    # torch version
    K = torch.cdist(
        torch.cartesian_prod(
            torch.linspace(dx/2, L - dx/2, n1), torch.linspace(dy/2, L - dy/2, n2)
        ).type(torch.DoubleTensor),
        torch.cartesian_prod(
            torch.linspace(dx/2, L - dx/2, n1), torch.linspace(dy/2, L - dy/2, n2)
        ).type(torch.DoubleTensor)) ** 2 /2
    K = torch.exp(-K / epsilon.cpu()) * cell_areas.cpu()**2  
    K_sum = (d.view(-1).cpu() - 1) @ (K @ (d.view(-1).cpu() - 1))

    assert torch.abs(dkd - K_sum) < 1e-9, f" dkd, K_sum = {dkd.item()}, {K_sum.item()}"

if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
