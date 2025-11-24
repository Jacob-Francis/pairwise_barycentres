import pytest
import torch
from mmuot import (
    alpha_reduction,
    generate_mmuotdataprocessor_star_graph,
    sinkhorn_update,
    mmuot_sinkhorn_loop,
    mmuot_marginals,
)
import numpy as np
import networkx as nx


@pytest.mark.parametrize(
    "n1, n2, L, grid_type",
    [
        (11, 11, 0.9, "flat"),
        (11, 10, 0.9, "flat"),
        (11, 12, 0.9, "flat"),
        (9, 8, 3.5, "tensor"),
        (8, 9, 3.5, "tensor"),
        (8, 8, 3.5, "tensor"),
        (12, 12, 6.0, "tuple"),
        (12, 13, 6.0, "tuple"),
        (12, 11, 6.0, "tuple"),
    ],
)  # noqa: E501
def test_sinkhorn_reduction_with_same_grid_uniform_density_uniform_measure_multi_it(
    n1, n2, L, grid_type
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

    # flat grid for truth
    Y = torch.cartesian_prod(torch.linspace(0, L, n1), torch.linspace(0, L, n2)).type(
        torch.DoubleTensor
    )

    data = []
    members = 3  # fits to compare against numpy

    for _ in range(members):
        data.append([None, None])  # uniform density, grid will equal everywhere

    # generate the barycentre dataprocessor class which will store all objects
    # of interest. It will also create the correct graph, and given no density of graphs
    # will create uniform densities on the grids
    dp = generate_mmuotdataprocessor_star_graph(data, grid=X, clear_grid=True)
    epsilon = dp._torch_numpy_process(L / np.sqrt(n1 * n2)).view(-1, 1)
    Y = dp._torch_numpy_process(Y)

    # Calculate f update using torhc first since sinkhorn_update
    # works in place

    # Calcuate projection
    # Torch version for comparison
    a_0_1_true = torch.exp(
        (
            dp.data_dict[1]["f"].view(-1, 1)
            - torch.cdist(
                Y,
                Y,
            )
            ** 2
            * dp.graph[0][1]["weight"]
            / 2
        )
        / epsilon
    ).sum(0) / np.prod(
        dp.data_dict[1]["f"].shape
    )  # times none since 2 is a leaf node
    # Torch version for comparison
    a_0_2_true = torch.exp(
        (
            dp.data_dict[2]["f"].view(-1, 1)
            - torch.cdist(
                Y,
                Y,
            )
            ** 2
            * dp.graph[0][2]["weight"]
            / 2
        )
        / epsilon
    ).sum(0) / np.prod(dp.data_dict[2]["f"].shape)

    f_0 = epsilon * torch.log(
        dp.data_dict[0]["density"].view(-1)
    ) - epsilon * torch.log(a_0_1_true.view(-1) * a_0_2_true.view(-1))

    # Check against what is should be
    for p_j, j in reversed(list(nx.dfs_tree(dp.graph, source=0).edges)):
        dp.data_dict[(p_j, j)]["alpha"] = alpha_reduction(
            dp, p_j, j, epsilon=epsilon, prod=False
        )

    f, err = sinkhorn_update(dp, 0, epsilon, rho=1.0, aprox="balanced", prod=False)
    dp.data_dict[0]["f"] = f.clone()
    assert torch.allclose(f_0.view(-1), f.view(-1), atol=1e-8), "Sinkhorn update failed"

    ################### Second it
    a_0_2_true = torch.exp(
        (
            dp.data_dict[2]["f"].view(-1, 1)
            - torch.cdist(
                Y,
                Y,
            )
            ** 2
            * dp.graph[0][2]["weight"]
            / 2
        )
        / epsilon
    ).sum(0) / np.prod(dp.data_dict[2]["f"].shape)

    alpha = alpha_reduction(dp, 0, 2, epsilon=epsilon, prod=False)

    assert torch.allclose(
        alpha.view(-1), a_0_2_true.view(-1), atol=1e-5
    ), "Alpha reduction recursion failed"

    a_1_0_true = (
        torch.exp(
            (
                dp.data_dict[0]["f"].view(-1, 1)
                - torch.cdist(
                    Y,
                    Y,
                )
                ** 2
                * dp.graph[0][1]["weight"]
                / 2
            )
            / epsilon
        )
        * a_0_2_true.view(-1, 1)
    ).sum(0) / np.prod(
        dp.data_dict[0]["f"].shape
    )  # times none since 2 is a leaf node

    alpha = alpha_reduction(dp, 1, 0, epsilon=epsilon, prod=False)

    assert torch.allclose(
        alpha.view(-1), a_1_0_true.view(-1), atol=1e-5
    ), "Alpha reduction recursion failed"

    f_2 = epsilon * torch.log(
        dp.data_dict[1]["density"].view(-1)
    ) - epsilon * torch.log(a_1_0_true.view(-1))

    # Check against what is should be
    for p_j, j in reversed(list(nx.dfs_tree(dp.graph, source=0).edges)):
        dp.data_dict[(p_j, j)]["alpha"] = alpha_reduction(
            dp, p_j, j, epsilon=epsilon, prod=False
        )

    # Check against what is should be
    for p_j, j in list(nx.dfs_tree(dp.graph, source=0).edges):
        dp.data_dict[(j, p_j)]["alpha"] = alpha_reduction(
            dp, j, p_j, epsilon=epsilon, prod=False
        )

    f, err = sinkhorn_update(dp, 1, epsilon, rho=1.0, aprox="balanced", prod=False)

    assert torch.allclose(f_2.view(-1), f.view(-1), atol=1e-8), "Sinkhorn update failed"


@pytest.mark.parametrize(
    "n1, n2, m1, m2, L, grid_type",
    [
        (11, 10, 5, 7, 0.9, "flat"),
        (8, 8, 13, 8, 3.5, "tensor"),
        (12, 11, 9, 9, 2.0, "tuple"),
    ],
)  # noqa: E501
def test_alpha_reduction_with_different_grid_random_density_prod_true(
    n1, n2, m1, m2, L, grid_type
):

    np.random.seed(n1 * n2 * m1 * m2)
    members = 2
    # tuple toggle for torch testing
    toggle = True
    if grid_type == "flat":
        data = []
        Y = torch.cartesian_prod(
            torch.linspace(0, L, m1), torch.linspace(0, L, m2)
        ).type(torch.DoubleTensor)
        density = torch.abs(torch.randn(m1 * m2))
        data.append([density, Y])  # central grid
        for m in range(members):  # member grids
            X = torch.cartesian_prod(
                torch.linspace(0, L, n1 + np.random.randint(-members, members)),
                torch.linspace(0, L, n2 + np.random.randint(-members, members)),
            ).type(torch.DoubleTensor)
            density = torch.abs(torch.randn_like(X[:, 0]))
            data.append([density, X])  # uniform density, grid will equal everywhere

    elif grid_type == "tensor":
        data = []
        Y = torch.stack(
            torch.meshgrid(
                torch.linspace(0, L, m1), torch.linspace(0, L, m2), indexing="ij"
            ),
            dim=-1,
        ).type(torch.DoubleTensor)
        density = torch.abs(torch.randn_like(Y[..., 0]))
        data.append([density, Y])  # central grid
        for m in range(members):
            X = torch.stack(
                torch.meshgrid(
                    torch.linspace(0, L, n1 + np.random.randint(-members, members)),
                    torch.linspace(0, L, n2 + np.random.randint(-members, members)),
                    indexing="ij",
                ),
                dim=-1,
            ).type(torch.DoubleTensor)
            density = torch.abs(torch.randn_like(X[..., 0]))
            data.append([density, X])

    elif grid_type == "tuple":
        toggle = False
        data = []
        Y = (torch.linspace(0, L, m1), torch.linspace(0, L, m2))
        density = torch.abs(torch.randn(m1, m2))
        data.append([density, Y])  # central grid
        for m in range(members):
            X = (
                torch.linspace(0, L, n1 + np.random.randint(-members, members)),
                torch.linspace(0, L, n2 + np.random.randint(-members, members)),
            )
            density = torch.abs(torch.randn(len(X[0]), len(X[1])))
            data.append([density, X])

    # generate the barycentre dataprocessor class which will store all objects
    # of interest. It will also create the correct graph, and given no density of graphs
    # will create uniform densities on the grids
    dp = generate_mmuotdataprocessor_star_graph(data, grid=None, clear_grid=False)
    epsilon = dp._torch_numpy_process(
        max(L / np.sqrt(n1 * n2), L / np.sqrt(m1 * m2))
    ).view(-1, 1)

    alpha = alpha_reduction(dp, 0, 1, epsilon=epsilon, prod=True)

    # Torch version for comparison
    a_0_1_true = (
        torch.exp(
            (
                dp.data_dict[1]["f"].view(-1, 1)
                - torch.cdist(
                    (
                        dp.data_dict[1]["grid"].view(-1, 2).to(alpha)
                        if toggle
                        else torch.cartesian_prod(*dp.data_dict[1]["grid"]).to(alpha)
                    ),
                    (
                        dp.data_dict[0]["grid"].view(-1, 2).to(alpha)
                        if toggle
                        else torch.cartesian_prod(*dp.data_dict[0]["grid"]).to(alpha)
                    ),
                )
                ** 2
                * dp.graph[0][1]["weight"]
                / 2
            )
            / epsilon
        )
        * dp.data_dict[1]["density"].view(-1, 1)
    ).sum(
        0
    )  # times none since 2 is a leaf node

    assert torch.allclose(
        alpha.view(-1), a_0_1_true.view(-1), atol=1e-8
    ), "Alpha reduction recursion failed"

    alpha = alpha_reduction(dp, 0, 2, epsilon=epsilon, prod=True)

    # Torch version for comparison
    a_0_2_true = (
        torch.exp(
            (
                dp.data_dict[2]["f"].view(-1, 1)
                - torch.cdist(
                    (
                        dp.data_dict[2]["grid"].view(-1, 2).to(alpha)
                        if toggle
                        else torch.cartesian_prod(*dp.data_dict[2]["grid"]).to(alpha)
                    ),
                    (
                        dp.data_dict[0]["grid"].view(-1, 2).to(alpha)
                        if toggle
                        else torch.cartesian_prod(*dp.data_dict[0]["grid"]).to(alpha)
                    ),
                )
                ** 2
                * dp.graph[0][2]["weight"]
                / 2
            )
            / epsilon
        )
        * dp.data_dict[2]["density"].view(-1, 1)
    ).sum(
        0
    )  # times none since 3 is a leaf node

    assert torch.allclose(
        alpha.view(-1), a_0_2_true.view(-1), atol=1e-8
    ), "Alpha reduction recursion failed"

    # Check sinkhorn
    f0 = -epsilon * torch.log(a_0_1_true.view(-1) * a_0_2_true.view(-1))
    f, err = sinkhorn_update(dp, 0, epsilon, rho=1.0, aprox="balanced", prod=True)
    dp.data_dict[0]["f"] = f.clone()
    assert torch.allclose(f0.view(-1), f.view(-1), atol=1e-8), "Sinkhorn update failed"

    # Update a_0_1
    alpha = alpha_reduction(dp, 0, 1, epsilon=epsilon, prod=True)

    # Torch version for comparison
    a_0_1_true = (
        torch.exp(
            (
                dp.data_dict[1]["f"].view(-1, 1)
                - torch.cdist(
                    (
                        dp.data_dict[1]["grid"].view(-1, 2).to(alpha)
                        if toggle
                        else torch.cartesian_prod(*dp.data_dict[1]["grid"]).to(alpha)
                    ),
                    (
                        dp.data_dict[0]["grid"].view(-1, 2).to(alpha)
                        if toggle
                        else torch.cartesian_prod(*dp.data_dict[0]["grid"]).to(alpha)
                    ),
                )
                ** 2
                * dp.graph[0][1]["weight"]
                / 2
            )
            / epsilon
        )
        * dp.data_dict[1]["density"].view(-1, 1)
    ).sum(
        0
    )  # times none since 2 is a leaf node

    assert torch.allclose(
        alpha.view(-1), a_0_1_true.view(-1), atol=1e-8
    ), "Alpha reduction recursion failed"

    # Torch version for comparison
    a_2_0_true = (
        (
            torch.exp(
                (
                    dp.data_dict[0]["f"].view(-1, 1)
                    - torch.cdist(
                        (
                            dp.data_dict[0]["grid"].view(-1, 2).to(alpha)
                            if toggle
                            else torch.cartesian_prod(*dp.data_dict[0]["grid"]).to(
                                alpha
                            )
                        ),
                        (
                            dp.data_dict[2]["grid"].view(-1, 2).to(alpha)
                            if toggle
                            else torch.cartesian_prod(*dp.data_dict[2]["grid"]).to(
                                alpha
                            )
                        ),
                    )
                    ** 2
                    * dp.graph[0][2]["weight"]
                    / 2
                )
                / epsilon
            )
            * a_0_1_true.view(-1, 1)
            * dp.data_dict[0]["density"].view(-1, 1)
        )
        .sum(0)
        .view(
            -1,
        )
    )
    alpha = alpha_reduction(dp, 2, 0, epsilon=epsilon, prod=True)

    assert torch.allclose(
        alpha.view(-1), a_2_0_true.view(-1), atol=1e-5
    ), "Alpha reduction recursion failed"

    # sinkhorn
    f2 = -epsilon * torch.log(a_2_0_true.view(-1))
    f, err = sinkhorn_update(dp, 2, epsilon, rho=1.0, aprox="balanced", prod=True)

    assert torch.allclose(f2.view(-1), f.view(-1), atol=1e-8), "Sinkhorn update failed"


# ------------------------------------------------------------------------------
#          TESTING MARGINALS CONVERGENCE
# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n1, n2, m1, m2, L, grid_type",
    [
        (11, 11, 11, 11, 1.0, "flat"),
        (11, 10, 5, 7, 0.9, "flat"),
        (9, 9, 12, 14, 1.0, "flat"),
        (8, 8, 13, 8, 3.5, "tensor"),
        (8, 8, 8, 12, 3.5, "tensor"),
        (12, 11, 9, 9, 2.0, "tuple"),
        (12, 11, 12, 11, 2.0, "tuple"),
    ],
)  # noqa: E501
def test_marginals_and_loop_uniform_density_uniform(n1, n2, m1, m2, L, grid_type):

    np.random.seed(n1 * n2 * m1 * m2)
    members = 2
    # tuple toggle for torch testing
    if grid_type == "flat":
        data = []
        Y = torch.cartesian_prod(
            torch.linspace(0, L, m1), torch.linspace(0, L, m2)
        ).type(torch.DoubleTensor)
        density = torch.abs(torch.randn(m1 * m2))
        data.append([None, Y])  # central grid
        for m in range(members):  # member grids
            X = torch.cartesian_prod(
                torch.linspace(0, L, n1 + np.random.randint(-members, members)),
                torch.linspace(0, L, n2 + np.random.randint(-members, members)),
            ).type(torch.DoubleTensor)
            density = torch.abs(torch.randn_like(X[:, 0]))
            data.append([None, X])  # uniform density, grid will equal everywhere

    elif grid_type == "tensor":
        data = []
        Y = torch.stack(
            torch.meshgrid(
                torch.linspace(0, L, m1), torch.linspace(0, L, m2), indexing="ij"
            ),
            dim=-1,
        ).type(torch.DoubleTensor)
        density = torch.abs(torch.randn_like(Y[..., 0]))
        data.append([None, Y])  # central grid
        for m in range(members):
            X = torch.stack(
                torch.meshgrid(
                    torch.linspace(0, L, n1 + np.random.randint(-members, members)),
                    torch.linspace(0, L, n2 + np.random.randint(-members, members)),
                    indexing="ij",
                ),
                dim=-1,
            ).type(torch.DoubleTensor)
            density = torch.abs(torch.randn_like(X[..., 0]))
            data.append([None, X])

    elif grid_type == "tuple":
        toggle = False
        data = []
        Y = (torch.linspace(0, L, m1), torch.linspace(0, L, m2))
        density = torch.abs(torch.randn(m1, m2))
        data.append([None, Y])  # central grid
        for m in range(members):
            X = (
                torch.linspace(0, L, n1 + np.random.randint(-members, members)),
                torch.linspace(0, L, n2 + np.random.randint(-members, members)),
            )
            density = torch.abs(torch.randn(len(X[0]), len(X[1])))
            data.append([None, X])

    # generate the barycentre dataprocessor class which will store all objects
    # of interest. It will also create the correct graph, and given no density of graphs
    # will create uniform densities on the grids
    dp = generate_mmuotdataprocessor_star_graph(data, grid=None, clear_grid=False)
    epsilon = dp._torch_numpy_process(
        max(L / np.sqrt(n1 * n2), L / np.sqrt(m1 * m2))
    ).view(-1, 1)

    dp = mmuot_sinkhorn_loop(
        dp,
        epsilon,
        rho=1.0,
        max_iterations=50,
        tol=1e-7,
        aprox="balanced",
        prod=False,
        convergence_tracking=False,
        verbose=False,
    )

    marginals, errors = mmuot_marginals(dp, epsilon, prod=False, alpha_update=False)

    print(errors)
    for k in errors:
        assert errors[k] < 1e-6, "Marginal did not converge sufficiently"


@pytest.mark.parametrize(
    "n1, n2, m1, m2, L, grid_type",
    [
        (43, 42, 44, 45, 1.0, "flat"),
        (50, 51, 52, 53, 1.0, "tuple"),
        (50, 50, 50, 54, 1.0, "tensor"),
        (50, 50, 50, 50, 1.0, "flat"),
        (43, 42, 44, 45, 1.0, "flat"),
    ],
)  # noqa: E501
def test_marginals_and_loop_random_density_uniformreference(
    n1, n2, m1, m2, L, grid_type
):

    np.random.seed(n1 * n2 * m1 * m2)
    members = 2
    # tuple toggle for torch testing
    if grid_type == "flat":
        data = []
        Y = torch.cartesian_prod(
            torch.linspace(0, L, m1), torch.linspace(0, L, m2)
        ).type(torch.DoubleTensor)
        density = torch.abs(torch.rand(m1 * m2))
        data.append([density / density.sum(), Y])  # central grid
        for m in range(members):  # member grids
            X = torch.cartesian_prod(
                torch.linspace(0, L, n1 + np.random.randint(-members, members)),
                torch.linspace(0, L, n2 + np.random.randint(-members, members)),
            ).type(torch.DoubleTensor)
            density = torch.abs(torch.rand_like(X[:, 0]))
            data.append(
                [density / density.sum(), X]
            )  # uniform density, grid will equal everywhere

    elif grid_type == "tensor":
        data = []
        Y = torch.stack(
            torch.meshgrid(
                torch.linspace(0, L, m1), torch.linspace(0, L, m2), indexing="ij"
            ),
            dim=-1,
        ).type(torch.DoubleTensor)
        density = torch.abs(torch.rand_like(Y[..., 0]))
        data.append([density / density.sum(), Y])  # central grid
        for m in range(members):
            X = torch.stack(
                torch.meshgrid(
                    torch.linspace(0, L, n1 + np.random.randint(-members, members)),
                    torch.linspace(0, L, n2 + np.random.randint(-members, members)),
                    indexing="ij",
                ),
                dim=-1,
            ).type(torch.DoubleTensor)
            density = torch.abs(torch.rand_like(X[..., 0]))
            data.append([density / density.sum(), X])

    elif grid_type == "tuple":
        toggle = False
        data = []
        Y = (torch.linspace(0, L, m1), torch.linspace(0, L, m2))
        density = torch.abs(torch.rand(m1, m2))
        data.append([density / density.sum(), Y])  # central grid
        for m in range(members):
            X = (
                torch.linspace(0, L, n1 + np.random.randint(-members, members)),
                torch.linspace(0, L, n2 + np.random.randint(-members, members)),
            )
            density = torch.abs(torch.rand(len(X[0]), len(X[1])))
            data.append([density / density.sum(), X])

    # generate the barycentre dataprocessor class which will store all objects
    # of interest. It will also create the correct graph, and given no density of graphs
    # will create uniform densities on the grids
    dp = generate_mmuotdataprocessor_star_graph(data, grid=None, clear_grid=False)
    epsilon = (
        dp._torch_numpy_process(max(L / np.sqrt(n1 * n2), L / np.sqrt(m1 * m2))).view(
            -1, 1
        )
        / 10
    )

    dp = mmuot_sinkhorn_loop(
        dp,
        epsilon,
        rho=1.0,
        max_iterations=600,
        tol=1e-12,
        aprox="balanced",
        prod=False,
        convergence_tracking=False,
        verbose=True,
    )

    marginals, errors = mmuot_marginals(dp, epsilon, prod=False, alpha_update=True)

    print(errors)
    for k in errors:
        # its very hard to get down to a sufficent level of convergnece in the margianls
        assert errors[k] < 1e-3, "Marginal did not converge sufficiently"


@pytest.mark.parametrize(
    "n1, n2, m1, m2, L, grid_type",
    [
        (20, 20, 20, 20, 1.0, "flat"),
        (50, 51, 52, 53, 1.0, "tuple"),
        (50, 50, 50, 54, 1.0, "tensor"),
    ],
)  # noqa: E501
def test_marginals_and_loop_random_density_product_reference(
    n1, n2, m1, m2, L, grid_type
):

    np.random.seed(n1 * n2 * m1 * m2)
    members = 2
    # tuple toggle for torch testing
    if grid_type == "flat":
        data = []
        Y = torch.cartesian_prod(
            torch.linspace(0, L, m1), torch.linspace(0, L, m2)
        ).type(torch.DoubleTensor)
        density = torch.abs(torch.rand(m1 * m2))
        data.append([density / density.sum(), Y])  # central grid
        for m in range(members):  # member grids
            X = torch.cartesian_prod(
                torch.linspace(0, L, n1 + np.random.randint(-members, members)),
                torch.linspace(0, L, n2 + np.random.randint(-members, members)),
            ).type(torch.DoubleTensor)
            density = torch.abs(torch.rand_like(X[:, 0]))
            data.append(
                [density / density.sum(), X]
            )  # uniform density, grid will equal everywhere

    elif grid_type == "tensor":
        data = []
        Y = torch.stack(
            torch.meshgrid(
                torch.linspace(0, L, m1), torch.linspace(0, L, m2), indexing="ij"
            ),
            dim=-1,
        ).type(torch.DoubleTensor)
        density = torch.abs(torch.rand_like(Y[..., 0]))
        data.append([density / density.sum(), Y])  # central grid
        for m in range(members):
            X = torch.stack(
                torch.meshgrid(
                    torch.linspace(0, L, n1 + np.random.randint(-members, members)),
                    torch.linspace(0, L, n2 + np.random.randint(-members, members)),
                    indexing="ij",
                ),
                dim=-1,
            ).type(torch.DoubleTensor)
            density = torch.abs(torch.rand_like(X[..., 0]))
            data.append([density / density.sum(), X])

    elif grid_type == "tuple":
        toggle = False
        data = []
        Y = (torch.linspace(0, L, m1), torch.linspace(0, L, m2))
        density = torch.abs(torch.rand(m1, m2))
        data.append([density / density.sum(), Y])  # central grid
        for m in range(members):
            X = (
                torch.linspace(0, L, n1 + np.random.randint(-members, members)),
                torch.linspace(0, L, n2 + np.random.randint(-members, members)),
            )
            density = torch.abs(torch.rand(len(X[0]), len(X[1])))
            data.append([density / density.sum(), X])

    # generate the barycentre dataprocessor class which will store all objects
    # of interest. It will also create the correct graph, and given no density of graphs
    # will create uniform densities on the grids
    dp = generate_mmuotdataprocessor_star_graph(data, grid=None, clear_grid=False)
    epsilon = (
        dp._torch_numpy_process(max(L / np.sqrt(n1 * n2), L / np.sqrt(m1 * m2))).view(
            -1, 1
        )
        / 10
    )

    dp = mmuot_sinkhorn_loop(
        dp,
        epsilon,
        rho=1.0,
        max_iterations=600,
        tol=1e-12,
        aprox="balanced",
        prod=True,
        convergence_tracking=False,
        verbose=False,
    )

    marginals, errors = mmuot_marginals(dp, epsilon, prod=True, alpha_update=True)

    for k in errors:
        # print(k, errors[k])
        # print(marginals[k].sum())
        # print(dp.data_dict[k]['density'].sum())
        # print(torch.norm(marginals[k]-dp.data_dict[k]['density'])/torch.norm(dp.data_dict[k]['density']))
        # print(torch.norm(marginals[k]-dp.data_dict[k]['density'], p=1)/torch.norm(dp.data_dict[k]['density'], p=1))
        # its very hard to get down to a sufficent level of convergnece in the margianls
        assert errors[k] < 1e-3, "Marginal did not converge sufficiently"


# @pytest.mark.parametrize(
#     "n1, n2, m1, m2, L, grid_type",
#     [
#         (20, 20, 20, 20, 1.0, "flat"),
#         (11, 10, 5, 7, 0.9, "flat"),
#         (9,9, 12,14, 1.0, "flat"),
#         (8, 8, 13, 8, 3.5, "tensor"),
#         (8, 8, 8,12, 3.5, "tensor"),
#         (12, 11, 9, 9, 2.0, "tuple"),
#         (12, 11, 12, 11, 2.0, "tuple"),
#     ],
# )  # noqa: E501
# def test_marginals_and_loop_random_density_prod(n1, n2, m1, m2, L, grid_type):

#     np.random.seed(n1*n2*m1*m2)
#     members = 2
#     # tuple toggle for torch testing
#     if grid_type == "flat":
#         data = []
#         Y = torch.cartesian_prod(
#             torch.linspace(0, L, m1), torch.linspace(0, L, m2)
#         ).type(torch.DoubleTensor)
#         density = torch.abs(torch.randn(m1*m2))
#         data.append([density/density.sum(), Y]) # central grid
#         for m in range(members): # member grids
#             X = torch.cartesian_prod(
#                 torch.linspace(0, L, n1+np.random.randint(-members, members)), torch.linspace(0, L, n2+np.random.randint(-members, members))
#             ).type(torch.DoubleTensor)
#             density = torch.abs(torch.randn_like(X[:,0]))
#             data.append([density/density.sum(), X])  # uniform density, grid will equal everywhere

#     elif grid_type == "tensor":
#         data = []
#         Y = torch.stack(
#             torch.meshgrid(
#                 torch.linspace(0, L, m1), torch.linspace(0, L, m2), indexing="ij"
#             ),
#             dim=-1,
#         ).type(torch.DoubleTensor)
#         density = torch.abs(torch.randn_like(Y[...,0]))
#         data.append([density/density.sum(), Y])  # central grid
#         for m in range(members):
#             X = torch.stack(
#                 torch.meshgrid(
#                     torch.linspace(0, L, n1+np.random.randint(-members, members)), torch.linspace(0, L, n2+np.random.randint(-members, members)), indexing="ij"
#                 ),
#                 dim=-1,
#             ).type(torch.DoubleTensor)
#             density = torch.abs(torch.randn_like(X[...,0]))
#             data.append([density/density.sum(), X])

#     elif grid_type == "tuple":
#         toggle = False
#         data = []
#         Y = (torch.linspace(0, L, m1), torch.linspace(0, L, m2))
#         density = torch.abs(torch.randn(m1, m2))
#         data.append([density/density.sum(), Y])  # central grid
#         for m in range(members):
#             X = (torch.linspace(0, L, n1+np.random.randint(-members, members)), torch.linspace(0, L, n2+np.random.randint(-members, members)))
#             density = torch.abs(torch.randn(len(X[0]), len(X[1])))
#             data.append([density/density.sum(), X])

#     # generate the barycentre dataprocessor class which will store all objects
#     # of interest. It will also create the correct graph, and given no density of graphs
#     # will create uniform densities on the grids
#     dp = generate_mmuotdataprocessor_star_graph(data, grid=None, clear_grid=False)
#     epsilon = dp._torch_numpy_process(max(L / np.sqrt(n1 * n2), L / np.sqrt(m1 * m2))).view(-1, 1)

#     dp = mmuot_sinkhorn_loop(dp,
#     epsilon/10,
#     rho=1.0,
#     max_iterations=100,
#     tol=1e-7,
#     aprox='balanced',
#     prod=True,
#     convergence_tracking=False,
#     verbose=True,
#     )

#     marginals, errors = mmuot_marginals(dp, epsilon, prod=True, alpha_update=False)

#     print(errors)
#     for k in errors:
#         assert errors[k] < 1e-6, "Marginal did not converge sufficiently"

# ------------------------------------------------------------------------------
#          TESTING COST CONVERGNCE?
# ------------------------------------------------------------------------------


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main([__file__]))
