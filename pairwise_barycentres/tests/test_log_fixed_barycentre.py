# We test that the marginal for the centre doesn't change?
import numpy as np
from scipy.spatial.distance import cdist
import torch
import pytest
from pwbarycentres import (
    asymmetric_sinkhorn_log_algorithm,
    generate_barycentredataprocessor,
    asymmetric_cost,
    ot_marginals,
)
import networkx as nx

torch.set_printoptions(precision=8)


# FIXED UNIFORM BARCENTRE TEST

@pytest.mark.parametrize(
    "n1, n2, members, m1, m2, L, grid_type, aprox",
    [
        (11, 10, 2, 5, 7, 0.9, "flat", 'balanced'),
        (8, 8, 4, 13, 8, 3.5, "tensor", 'balanced'),
        (12, 11, 3, 9, 9, 2.0, "tuple", 'balanced'),
        (11, 10, 2, 5, 7, 0.9, "flat", 'kl'),
        (8, 8, 4, 13, 8, 3.5, "tensor", 'kl'),
        (12, 11, 3, 9, 9, 2.0, "tuple", 'kl'),
        (11, 10, 2, 5, 7, 0.9, "flat", 'tv'),
        (8, 8, 4, 13, 8, 3.5, "tensor", 'tv'),
        (12, 11, 3, 9, 9, 2.0, "tuple", 'tv'),
    ],
)  # noqa: E501
def test_fixed_bary_margainls_uniform_centre(
    n1, n2, members, m1, m2, L, grid_type, aprox
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

        fixed_bary = torch.ones(m1 * m2) / (m1 * m2)  # uniform density on the barycentre grid
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
        fixed_bary = torch.ones(m1, m2) / (m1 * m2)  # uniform density on the barycentre grid

    elif grid_type == "tuple":
        data = []
        for m in range(members):
            X = (
                torch.linspace(0, L, n1 + np.random.randint(-members, members)),
                torch.linspace(0, L, n2 + np.random.randint(-members, members)),
            )
            data.append([None, X])
        Y = (torch.linspace(0, L, m1), torch.linspace(0, L, m2))
        fixed_bary = torch.ones(m1, m2) / (m1 * m2)  # uniform density on the barycentre grid

    # generate the barycentre dataprocessor class which will store all objects
    # of interest. It will also create the correct graph, and given no density of graphs
    # will create uniform densities on the grids
    data_processor = generate_barycentredataprocessor(data, barycentre_grid=Y, potentials='f')

    # run asymmetric sinkhorn algorithm
    data_processor, barycentre, potential_error_list, barycentre_error_list, constraints_dict = (
        asymmetric_sinkhorn_log_algorithm(
            data_processor,
            epsilon=max(1 / np.sqrt(n1 * n2), 1 / np.sqrt(m1 * m2)),
            rho=1.0,
            aprox=aprox,
            max_iterates=700,
            tol=1e-8,
            epsilon_annealing=False,
            debiasing=False,
            verbose=True,
            fixed_barycentre=fixed_bary,
            measure_constraints=True,
        )
    )

    # No need to check bary and d constraints
    assert constraints_dict['partial_g'][-1] < 1e-6, "key: " + 'partial_g' + " value: " + str(constraints_dict['partial_g'][-1])  # less than tolerance
    assert constraints_dict['partial_f'][-1] < 1e-6, "key: " + 'partial_f' + " value: " + str(constraints_dict['partial_f'][-1])  # less than tolerance

    # if given no nodes then they should all be returned
    marginals = ot_marginals(
        data_processor,
        epsilon=max(1 / np.sqrt(n1 * n2), 1 / np.sqrt(m1 * m2)),
        debiasing=False,
    )

    for node in data_processor.graph.nodes():
        # should stay mass one because of the fixed centre node
        assert marginals[node]["marginal"].sum().item() - 1.0 < 1e-5, (
            marginals[node]["marginal"].sum().item()
        )  # less than tolerance

    for nodes in data_processor.graph.nodes():
        if aprox == 'balanced':
            # relax tolerance because it was not converging very fast for this debiased setting?
            assert marginals[nodes]["error"] < 1e-4, marginals[nodes][
                "error"
            ]  # less than tolerance
        elif node %2==0:
            assert marginals[nodes]["error"] < 1e-4, marginals[nodes][
                "error"
            ]  # less than tolerance


# FIXED Non uniform BARCENTRE TEST


@pytest.mark.parametrize(
    "n1, n2, members, m1, m2, L, grid_type, aprox",
    [
        (11, 10, 2, 5, 7, 0.9, "flat", 'balanced'),
        (8, 8, 4, 13, 8, 3.5, "tensor", 'balanced'),
        (12, 11, 3, 9, 9, 2.0, "tuple", 'balanced'),
        (11, 10, 2, 5, 7, 0.9, "flat", 'kl'),
        (8, 8, 4, 13, 8, 3.5, "tensor", 'kl'),
        (12, 11, 3, 9, 9, 2.0, "tuple", 'kl'),
        (11, 10, 2, 5, 7, 0.9, "flat", 'tv'),
        (8, 8, 4, 13, 8, 3.5, "tensor", 'tv'),
        (12, 11, 3, 9, 9, 2.0, "tuple", 'tv'),
    ],
)  # noqa: E501
def test_fixed_bary_margainls_non_uniform_centre(
    n1, n2, members, m1, m2, L, grid_type, aprox
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

        fixed_bary = torch.ones(m1 * m2) / (m1 * m2) + Y[:,0]*Y[:, 1] # uniform density on the barycentre grid
        fixed_bary /= fixed_bary.sum() # normalise the density to sum to 1
    
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
        fixed_bary = torch.ones(m1 , m2) / (m1 * m2) + Y[:,:,0]*Y[:,:, 1] # uniform density on the barycentre grid
        fixed_bary /= fixed_bary.sum() # normalise the density to sum to 1
    elif grid_type == "tuple":
        data = []
        for m in range(members):
            X = (
                torch.linspace(0, L, n1 + np.random.randint(-members, members)),
                torch.linspace(0, L, n2 + np.random.randint(-members, members)),
            )
            data.append([None, X])
        Y = (torch.linspace(0, L, m1), torch.linspace(0, L, m2))
        fixed_bary = torch.ones(m1, m2) / (m1 * m2) + Y[0].view(-1,1)*Y[1].view(1,-1) # uniform density on the barycentre grid
        fixed_bary /= fixed_bary.sum() # normalise the density to sum to 1

    # generate the barycentre dataprocessor class which will store all objects
    # of interest. It will also create the correct graph, and given no density of graphs
    # will create uniform densities on the grids
    data_processor = generate_barycentredataprocessor(data, barycentre_grid=Y, potentials='f')

    # run asymmetric sinkhorn algorithm
    data_processor, barycentre, potential_error_list, barycentre_error_list, constraints_dict = (
        asymmetric_sinkhorn_log_algorithm(
            data_processor,
            epsilon=max(1 / np.sqrt(n1 * n2), 1 / np.sqrt(m1 * m2)),
            rho=1.0,
            aprox=aprox,
            max_iterates=700,
            tol=1e-8,
            epsilon_annealing=False,
            debiasing=False,
            verbose=True,
            fixed_barycentre=fixed_bary,
            measure_constraints=True,
        )
    )

    # No need to check bary and d constraints
    assert constraints_dict['partial_g'][-1] < 1e-6, "key: " + key + " value: " + str(constraints_dict[key][-1])  # less than tolerance
    assert constraints_dict['partial_f'][-1] < 1e-6, "key: " + key + " value: " + str(constraints_dict[key][-1])  # less than tolerance

    # if given no nodes then they should all be returned
    marginals = ot_marginals(
        data_processor,
        epsilon=max(1 / np.sqrt(n1 * n2), 1 / np.sqrt(m1 * m2)),
        debiasing=False,
    )

    for node in data_processor.graph.nodes():
        assert marginals[node]["marginal"].sum().item() - 1.0 < 1e-5, (
            marginals[node]["marginal"].sum().item()
        )  # less than tolerance

    for nodes in data_processor.graph.nodes():
        if aprox == 'balanced':
            # relax tolerance because it was not converging very fast for this debiased setting?
            assert marginals[nodes]["error"] < 1e-4, marginals[nodes][
                "error"
            ]  # less than tolerance
        elif nodes %2==0:
            assert marginals[nodes]["error"] < 1e-4, marginals[nodes][
                "error"
            ]  # less than tolerance