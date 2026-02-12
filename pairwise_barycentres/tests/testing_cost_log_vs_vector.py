import numpy as np
from scipy.spatial.distance import cdist
import torch
import pytest
from pwbarycentres import (
    asymmetric_sinkhorn_algorithm,
    generate_barycentredataprocessor,
    asymmetric_cost,
    asymmetric_sinkhorn_log_algorithm,
    ot_marginals,
)
import networkx as nx


# -------------------------------------------------------------------------------------------
# Testing all on different grids (though still same tpye for now)
# -------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n1, n2, members, m1, m2, L, grid_type, debiasing, aprox",
    [
        (11, 10, 3, 5, 7, 0.9, "flat", True, "balanced"),
        (8, 8, 4, 13, 8, 3.5, "tensor", True, "balanced"),
        (12, 11, 3, 9, 9, 2.0, "tuple", True, "balanced"),
    ],
)  # noqa: E501
def test_cost_vs_with_all_different_grids_with_debiasing(
    n1, n2, members, m1, m2, L, grid_type, debiasing, aprox
):

    np.random.seed(n1 * n2 * members * m1 * m2)

    if grid_type == "flat":
        data = []
        for m in range(members):
            X = torch.cartesian_prod(
                torch.linspace(0, L, n1 + np.random.randint(-members, members)),
                torch.linspace(0, L, n2 + np.random.randint(-members, members)),
            ).type(torch.DoubleTensor)
            data.append([None, X])  # uniform density, grid will equal everywhere

        Y = torch.cartesian_prod(
            torch.linspace(0, L, m1), torch.linspace(0, L, m2)
        ).type(torch.DoubleTensor)
    elif grid_type == "tensor":
        data = []
        for m in range(members):
            X = torch.stack(
                torch.meshgrid(
                    torch.linspace(0, L, n1 + np.random.randint(-members, members)),
                    torch.linspace(0, L, n2 + np.random.randint(-members, members)),
                    indexing="ij",
                ),
                dim=-1,
            ).type(torch.DoubleTensor)
            data.append([None, X])
        Y = torch.stack(
            torch.meshgrid(
                torch.linspace(0, L, m1), torch.linspace(0, L, m2), indexing="ij"
            ),
            dim=-1,
        ).type(torch.DoubleTensor)
    elif grid_type == "tuple":
        data = []
        for m in range(members):
            X = (
                torch.linspace(0, L, n1 + np.random.randint(-members, members)),
                torch.linspace(0, L, n2 + np.random.randint(-members, members)),
            )
            data.append([None, X])
        Y = (torch.linspace(0, L, m1), torch.linspace(0, L, m2))

    # LOG VERSION
    data_processor_log = generate_barycentredataprocessor(
        data, 
        barycentre_grid=Y, 
        potentials='f'
        )

    # run asymmetric sinkhorn algorithm
    data_processor_log, barycentre, potential_error_list, barycentre_error_list = (
        asymmetric_sinkhorn_log_algorithm(
            data_processor_log,
            epsilon=max(1 / np.sqrt(n1 * n2), 1 / np.sqrt(m1 * m2)),
            rho=1.0,
            aprox=aprox,
            max_iterates=1000,
            tol=1e-9,
            epsilon_annealing=True,
            debiasing=debiasing,
            verbose=True,
        )
    )

    cost_log, cost_list_log, dict_log = asymmetric_cost(
        data_processor_log, 
        epsilon=max(1 / np.sqrt(n1 * n2), 1 / np.sqrt(m1 * m2)), 
        rho=1.0, 
        aprox=aprox,
        debiasing=debiasing,
        return_breakdown=True
    )

    # VECTOR VERSION
    data_processor_vec = generate_barycentredataprocessor(
        data, 
        barycentre_grid=Y, 
        potentials='a'
        )

    # run asymmetric sinkhorn algorithm
    data_processor_vec, barycentre, potential_error_list, barycentre_error_list = (
        asymmetric_sinkhorn_algorithm(
            data_processor_vec,
            epsilon=max(1 / np.sqrt(n1 * n2), 1 / np.sqrt(m1 * m2)),
            rho=1.0,
            aprox=aprox,
            max_iterates=1000,
            tol=1e-9,
            epsilon_annealing=False,
            debiasing=debiasing,
            verbose=True,
        )
    )

    cost_vec, cost_list_vec, dict_vec = asymmetric_cost(
        data_processor_vec, 
        epsilon=max(1 / np.sqrt(n1 * n2), 1 / np.sqrt(m1 * m2)), 
        rho=1.0, 
        aprox=aprox,
        debiasing=debiasing,
        return_breakdown=True
    )

    # test if theyre close
    tol = 1e-3
    assert abs(cost_log - cost_vec) < 1e-5, f"Costs differ by {abs(cost_log - cost_vec).item()}, log: {cost_log.item()}, vec: {cost_vec.item()}"
    for log_term, vec_term in zip(dict_log['unbalanced_sinkhorn_terms'], dict_vec['unbalanced_sinkhorn_terms']):
        if type(log_term) == list:
            for lt, vt in zip(log_term, vec_term):
                assert abs(lt - vt) < 1e-5, f"Terms differ by {abs(lt - vt)}"
        else:
            assert abs(log_term - vec_term) < 1e-5, f"Terms differ by {abs(log_term - vec_term)}"


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)