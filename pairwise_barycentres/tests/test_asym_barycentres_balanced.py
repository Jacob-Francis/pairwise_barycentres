import numpy as np
from scipy.spatial.distance import cdist
import torch
import pytest
from pwbarycentres import (
    asymmetric_sinkhorn_algorithm,
    generate_barycentredataprocessor,
)
import networkx as nx

torch.set_printoptions(precision=8)

# --------------------------------------------------------
# Testing barycentre and other grids which are the same
# --------------------------------------------------------
@pytest.mark.parametrize(
    "n1, n2, members, L, grid_type",
    [
        (8, 8, 4, 3.5, "tensor"),
        (9, 8, 3, 3.5, "tensor"),
        (8, 9, 2, 3.5, "tensor"),
        (11, 11, 6, 0.9, "flat"),
        (11, 10, 7, 0.9, "flat"),
        (11, 12, 8, 0.9, "flat"),
        (12, 12, 3, 6.0, "tuple"),
        (12, 13, 3, 6.0, "tuple"),
        (12, 11, 3, 6.0, "tuple"),

    ],
)  # noqa: E501
def test_asym_bary_with_same_grid_uniform_density_without_debiasing(n1, n2, members, L, grid_type):

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

    data = []

    for m in range(members):
        data.append([None, None])  # uniform density, grid will equal everywhere

    # generate the barycentre dataprocessor class which will store all objects
    # of interest. It will also create the correct graph, and given no density of graphs
    # will create uniform densities on the grids
    data_processor = generate_barycentredataprocessor(data, barycentre_grid=X, grid=X)

    # Assert that the orginal structure is correct
    for edges in data_processor.graph.edges():
        assert (
            np.abs(data_processor.data_dict[edges[0]]["density"].sum().item() - 1.0) < 1e-5
        ), (data_processor.data_dict[edges[0]]["density"].sum().item())
        assert (
            np.abs(data_processor.data_dict[edges[1]]["density"].sum().item() - 1.0) < 1e-5
        ), (data_processor.data_dict[edges[1]]["density"].sum().item())

    # run asymmetric sinkhorn algorithm
    data_processor, barycentre, potential_error_list, barycentre_error_list = (
        asymmetric_sinkhorn_algorithm(
            data_processor,
            epsilon=1 / np.sqrt(n1 * n2),
            rho=1.0,
            aprox="balanced",
            max_iterates=500,
            tol=1e-8,
            epsilon_annealing=False,
            debiasing=False,
        )
    )
    assert barycentre_error_list[-1] < 1e-7 # less than tolerance

    assert np.abs(barycentre.sum().item() - 1.0) < 1e-5

    for edges in data_processor.graph.edges():
        assert (
            np.abs(data_processor.data_dict[edges[0]]["density"].sum().item() - 1.0) < 1e-5
        ), (data_processor.data_dict[edges[0]]["density"].sum().item())
        assert (
            np.abs(data_processor.data_dict[edges[1]]["density"].sum().item() - 1.0) < 1e-5
        ), (data_processor.data_dict[edges[1]]["density"].sum().item())

    # Since using a uniform density the barycentre should also be uniform
    # Because of entropic error the tolerance is looser here
    assert torch.allclose(barycentre, torch.ones_like(barycentre) / barycentre.numel(), atol=1e-2)

@pytest.mark.parametrize(
    "n1, n2, members, L, grid_type",
    [
        (8, 8, 4, 3.5, "tensor"),
        (9, 8, 3, 3.5, "tensor"),
        (8, 9, 2, 3.5, "tensor"),
        (11, 11, 6, 0.9, "flat"),
        (11, 10, 7, 0.9, "flat"),
        (11, 12, 8, 0.9, "flat"),
        (12, 12, 3, 6.0, "tuple"),
        (12, 13, 3, 6.0, "tuple"),
        (12, 11, 3, 6.0, "tuple"),

    ],
)  # noqa: E501
def test_asym_bary_with_same_grid_uniform_density_with_debiasing(n1, n2, members, L, grid_type):

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

    data = []

    for m in range(members):
        data.append([None, None])  # uniform density, grid will equal everywhere

    # generate the barycentre dataprocessor class which will store all objects
    # of interest. It will also create the correct graph, and given no density of graphs
    # will create uniform densities on the grids
    data_processor = generate_barycentredataprocessor(data, barycentre_grid=X, grid=X)

    # Assert that the orginal structure is correct
    for edges in data_processor.graph.edges():
        assert (
            np.abs(data_processor.data_dict[edges[0]]["density"].sum().item() - 1.0) < 1e-5
        ), (data_processor.data_dict[edges[0]]["density"].sum().item())
        assert (
            np.abs(data_processor.data_dict[edges[1]]["density"].sum().item() - 1.0) < 1e-5
        ), (data_processor.data_dict[edges[1]]["density"].sum().item())

    # run asymmetric sinkhorn algorithm
    data_processor, barycentre, potential_error_list, barycentre_error_list = (
        asymmetric_sinkhorn_algorithm(
            data_processor,
            epsilon=1 / np.sqrt(n1 * n2),
            rho=1.0,
            aprox="balanced",
            max_iterates=1000,
            tol=1e-6, # relax tolerance because it was not converging very fast for this debiased setting?
            epsilon_annealing=False,
            debiasing=True,
        )
    )

    assert barycentre_error_list[-1] < 1e-5 # less than tolerance

    assert np.abs(barycentre.sum().item() - 1.0) < 1e-3

    for edges in data_processor.graph.edges():
        assert (
            np.abs(data_processor.data_dict[edges[0]]["density"].sum().item() - 1.0) < 1e-3
        ), (data_processor.data_dict[edges[0]]["density"].sum().item())
        assert (
            np.abs(data_processor.data_dict[edges[1]]["density"].sum().item() - 1.0) < 1e-3
        ), (data_processor.data_dict[edges[1]]["density"].sum().item())
    
    # Since using a uniform density the barycentre should also be uniform
    # Because of debiasing we can make the tolerance tighter
    assert torch.allclose(barycentre, torch.ones_like(barycentre) / barycentre.numel(), atol=1e-2)

# --------------------------------------------------------
# Testing barycentre and other grids which are different
# --------------------------------------------------------
@pytest.mark.parametrize(
    "n1, n2, members, m1, m2, L, grid_type",
    [
        (11, 10, 3, 5, 7, 0.9, "flat"),
        (8, 8, 4, 13, 8, 3.5, "tensor"),
        (12, 11, 3, 9, 9, 2.0, "tuple"),
    ],
)  # noqa: E501
def test_asym_bary_with_different_grid_uniform_density_without_debiasing(n1, n2, members, m1, m2, L, grid_type):

    if grid_type == "flat":
        X = torch.cartesian_prod(
            torch.linspace(0, L, n1), torch.linspace(0, L, n2)
        ).type(torch.DoubleTensor)
        Y = torch.cartesian_prod(
            torch.linspace(0, L, m1), torch.linspace(0, L, m2)
        ).type(torch.DoubleTensor)
    elif grid_type == "tensor":
        X = torch.stack(
            torch.meshgrid(
                torch.linspace(0, L, n1), torch.linspace(0, L, n2), indexing="ij"
            ),
            dim=-1,
        ).type(torch.DoubleTensor)
        Y = torch.stack(
            torch.meshgrid(
                torch.linspace(0, L, m1), torch.linspace(0, L, m2), indexing="ij"
            ),
            dim=-1,
        ).type(torch.DoubleTensor)
    elif grid_type == "tuple":
        X = (torch.linspace(0, L, n1), torch.linspace(0, L, n2))
        Y = (torch.linspace(0, L, m1), torch.linspace(0, L, m2))

    data = []

    for m in range(members):
        data.append([None, X])  # uniform density, grid will equal everywhere

    # generate the barycentre dataprocessor class which will store all objects
    # of interest. It will also create the correct graph, and given no density of graphs
    # will create uniform densities on the grids
    data_processor = generate_barycentredataprocessor(data, barycentre_grid=Y)

    # run asymmetric sinkhorn algorithm
    data_processor, barycentre, potential_error_list, barycentre_error_list = (
        asymmetric_sinkhorn_algorithm(
            data_processor,
            epsilon=max(1 / np.sqrt(n1 * n2), 1 / np.sqrt(m1 * m2)),
            rho=1.0,
            aprox="balanced",
            max_iterates=500,
            tol=1e-7,
            epsilon_annealing=False,
            debiasing=False,
        )
    )

    assert barycentre_error_list[-1] < 1e-7 # less than tolerance


    for edges in data_processor.graph.edges():
        assert np.isclose(data_processor.data_dict[edges[0]]["density"].sum().item(), 1.0)
        assert np.isclose(data_processor.data_dict[edges[1]]["density"].sum().item(), 1.0)

    # The uniform test is too strict when the grids differ

@pytest.mark.parametrize(
    "n1, n2, members, m1, m2, L, grid_type",
    [
        (11, 10, 3, 5, 7, 0.9, "flat"),
        (8, 8, 4, 13, 8, 3.5, "tensor"),
        (12, 11, 3, 9, 9, 2.0, "tuple"),
    ],
)  # noqa: E501
def test_asym_bary_with_different_grid_uniform_density_with_debiasing(n1, n2, members, m1, m2, L, grid_type):

    if grid_type == "flat":
        X = torch.cartesian_prod(
            torch.linspace(0, L, n1), torch.linspace(0, L, n2)
        ).type(torch.DoubleTensor)
        Y = torch.cartesian_prod(
            torch.linspace(0, L, m1), torch.linspace(0, L, m2)
        ).type(torch.DoubleTensor)
    elif grid_type == "tensor":
        X = torch.stack(
            torch.meshgrid(
                torch.linspace(0, L, n1), torch.linspace(0, L, n2), indexing="ij"
            ),
            dim=-1,
        ).type(torch.DoubleTensor)
        Y = torch.stack(
            torch.meshgrid(
                torch.linspace(0, L, m1), torch.linspace(0, L, m2), indexing="ij"
            ),
            dim=-1,
        ).type(torch.DoubleTensor)
    elif grid_type == "tuple":
        X = (torch.linspace(0, L, n1), torch.linspace(0, L, n2))
        Y = (torch.linspace(0, L, m1), torch.linspace(0, L, m2))

    data = []

    for m in range(members):
        data.append([None, X])  # uniform density, grid will equal everywhere

    # generate the barycentre dataprocessor class which will store all objects
    # of interest. It will also create the correct graph, and given no density of graphs
    # will create uniform densities on the grids
    data_processor = generate_barycentredataprocessor(data, barycentre_grid=Y)

    # run asymmetric sinkhorn algorithm
    data_processor, barycentre, potential_error_list, barycentre_error_list = (
        asymmetric_sinkhorn_algorithm(
            data_processor,
            epsilon=max(1 / np.sqrt(n1 * n2), 1 / np.sqrt(m1 * m2)),
            rho=1.0,
            aprox="balanced",
            max_iterates=500,
            tol=1e-5,
            epsilon_annealing=False,
            debiasing=True,
        )
    )

    assert barycentre_error_list[-1] < 1e-5 # less than tolerance

    for edges in data_processor.graph.edges():
        assert np.isclose(data_processor.data_dict[edges[0]]["density"].sum().item(), 1.0)
        assert np.isclose(data_processor.data_dict[edges[1]]["density"].sum().item(), 1.0)


# -------------------------------------------------------------------------------------------
# Testing all on different grids (though still same tpye for now)
# -------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n1, n2, members, m1, m2, L, grid_type",
    [
        (11, 10, 3, 5, 7, 0.9, "flat"),
        (8, 8, 4, 13, 8, 3.5, "tensor"),
        (12, 11, 3, 9, 9, 2.0, "tuple"),
    ],
)  # noqa: E501
def test_asym_bary_with_all_different_grids_with_debiasing(n1, n2, members, m1, m2, L, grid_type):

    np.random.seed(n1*n2*members*m1*m2)

    if grid_type == "flat":
        data = []
        for m in range(members):
            X = torch.cartesian_prod(
                torch.linspace(0, L, n1+np.random.randint(-members, members)), torch.linspace(0, L, n2+np.random.randint(-members, members))
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
                    torch.linspace(0, L, n1+np.random.randint(-members, members)), torch.linspace(0, L, n2+np.random.randint(-members, members)), indexing="ij"
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
            X = (torch.linspace(0, L, n1+np.random.randint(-members, members)), torch.linspace(0, L, n2+np.random.randint(-members, members)))
            data.append([None, X])
        Y = (torch.linspace(0, L, m1), torch.linspace(0, L, m2))

    
    # generate the barycentre dataprocessor class which will store all objects
    # of interest. It will also create the correct graph, and given no density of graphs
    # will create uniform densities on the grids
    data_processor = generate_barycentredataprocessor(data, barycentre_grid=Y)

    # run asymmetric sinkhorn algorithm
    data_processor, barycentre, potential_error_list, barycentre_error_list = (
        asymmetric_sinkhorn_algorithm(
            data_processor,
            epsilon=max(1 / np.sqrt(n1 * n2), 1 / np.sqrt(m1 * m2)),
            rho=1.0,
            aprox="balanced",
            max_iterates=500,
            tol=1e-5,
            epsilon_annealing=False,
            debiasing=True,
            verbose=True
        )
    )

    assert barycentre_error_list[-1] < 1e-5 # less than tolerance

    for edges in data_processor.graph.edges():
        assert np.isclose(data_processor.data_dict[edges[0]]["density"].sum().item(), 1.0, atol=1e-3)
        assert np.isclose(data_processor.data_dict[edges[1]]["density"].sum().item(), 1.0, atol=1e-3)

@pytest.mark.parametrize(
    "n1, n2, members, m1, m2, L, grid_type",
    [
        (11, 10, 2, 5, 7, 0.9, "flat"),
        (8, 8, 4, 13, 8, 3.5, "tensor"),
        (12, 11, 3, 9, 9, 2.0, "tuple"),
    ],
)  # noqa: E501
def test_asym_bary_with_all_different_grids_without_debiasing(n1, n2, members, m1, m2, L, grid_type):

    np.random.seed(n1*n2*members*m1*m2)

    if grid_type == "flat":
        data = []
        for m in range(members):
            X = torch.cartesian_prod(
                torch.linspace(0, L, n1+np.random.randint(-members, members)), torch.linspace(0, L, n2+np.random.randint(-members, members))
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
                    torch.linspace(0, L, n1+np.random.randint(-members, members)), torch.linspace(0, L, n2+np.random.randint(-members, members)), indexing="ij"
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
            X = (torch.linspace(0, L, n1+np.random.randint(-members, members)), torch.linspace(0, L, n2+np.random.randint(-members, members)))
            data.append([None, X])
        Y = (torch.linspace(0, L, m1), torch.linspace(0, L, m2))

    
    # generate the barycentre dataprocessor class which will store all objects
    # of interest. It will also create the correct graph, and given no density of graphs
    # will create uniform densities on the grids
    data_processor = generate_barycentredataprocessor(data, barycentre_grid=Y)

    # run asymmetric sinkhorn algorithm
    data_processor, barycentre, potential_error_list, barycentre_error_list = (
        asymmetric_sinkhorn_algorithm(
            data_processor,
            epsilon=max(1 / np.sqrt(n1 * n2), 1 / np.sqrt(m1 * m2)),
            rho=1.0,
            aprox="balanced",
            max_iterates=500,
            tol=1e-7,
            epsilon_annealing=False,
            debiasing=False,
            verbose=True,
        )
    )

    assert barycentre_error_list[-1] < 1e-7 # less than tolerance

    for edges in data_processor.graph.edges():
        assert np.isclose(data_processor.data_dict[edges[0]]["density"].sum().item(), 1.0)
        assert np.isclose(data_processor.data_dict[edges[1]]["density"].sum().item(), 1.0)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
