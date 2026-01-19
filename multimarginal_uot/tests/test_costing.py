import pytest
import torch
from mmuot import (
    alpha_reduction,
    generate_mmuotdataprocessor_star_graph,
    sinkhorn_update,
    mmuot_sinkhorn_loop,
    mmuot_marginals,
    mmuot_dual_cost , 
    kernel_size,   
)
import numpy as np
import networkx as nx


# ------------------------------------------------------------------------------
#         TESTING COSTING AGAINST NUMPY IMPLEMENTATION
# ------------------------------------------------------------------------------

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
def test_costing_with_same_grid_uniform_density_uniform_measure_multi_it(
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
    ) - epsilon * torch.log(a_0_1_true.view(-1) * a_0_2_true.view(-1) / np.prod(dp.data_dict[0]["f"].shape))

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
    ) - epsilon * torch.log(a_1_0_true.view(-1)/ np.prod(dp.data_dict[2]["f"].shape))

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
    dp.data_dict[1]["f"] = f.clone()

    assert torch.allclose(f_2.view(-1), f.view(-1), atol=1e-8), "Sinkhorn update failed"

    # calcaute mm cost
    cost = mmuot_dual_cost(dp, epsilon, rho=1.0, aprox="balanced", prod=False, no_kernal_term=True)

    # Numpy version - only calculated f0 and f2; f1 is zero
    dual_cost = 0.0
    dual_cost += f_0.squeeze().dot(dp.data_dict[0]['density'].view(-1))
    dual_cost += f_2.squeeze().dot(dp.data_dict[2]['density'].view(-1))

    dual_cost -= epsilon.squeeze()*(a_1_0_true.squeeze()*dp.data_dict[1]['density'].view(-1)).sum(0)

    assert np.isclose(cost.cpu().numpy(), dual_cost.cpu().numpy(), atol=1e-8), "Dual costing failed"


@pytest.mark.parametrize(
    "n1, n2, m1, m2, L, grid_type",
    [
        (11, 10, 5, 7, 0.9, "flat"),
        (8, 8, 13, 8, 3.5, "tensor"),
        (12, 11, 9, 9, 2.0, "tuple"),
    ],
)  # noqa: E501
def test_costing_with_different_grid_random_density_prod_true(
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
    dp.data_dict[2]["f"] = f.clone()

    assert torch.allclose(f2.view(-1), f.view(-1), atol=1e-8), "Sinkhorn update failed"

    # calcaute mm cost
    cost = mmuot_dual_cost(dp, epsilon, rho=1.0, aprox="balanced", prod=True, no_kernal_term=True)

    # Numpy version - only calculated f0 and f2; f1 is zero
    dual_cost = 0.0
    dual_cost += f0.squeeze().dot(dp.data_dict[0]['density'].view(-1))
    dual_cost += f2.squeeze().dot(dp.data_dict[2]['density'].view(-1))
    
    dual_cost -= epsilon.squeeze()*(a_2_0_true.squeeze()*dp.data_dict[2]['density'].view(-1)).sum()

    print(cost.cpu().numpy(), dual_cost.cpu().numpy())
    assert np.isclose(cost.cpu().numpy(), dual_cost.cpu().numpy(), atol=1e-8), "Dual costing failed"


# ------------------------------------------------------------------------------
#         TESTING KERNEL REDUCTION
# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n1, n2, m1, m2, L, grid_type",
    [
        (11, 10, 5, 7, 0.9, "flat"),
        (8, 8, 13, 8, 3.5, "tensor"),
        (12, 11, 9, 9, 2.0, "tuple"),
    ],
)  # noqa: E501
def test_kernel_with_different_grid_random_density_prod_true(
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

    # kernal size reduction 
    kernel_sum = kernel_size(dp, epsilon, prod=True)

    alpha = alpha_reduction(dp, 0, 1, epsilon=epsilon, prod=True)

    # Torch version for comparison
    c_0_1_true = (
        torch.exp(
            (
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
    )

    # Torch version for comparison
    c_0_2_true = (
        torch.exp(
            (
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
    )

    # Need some sort of 3D broadcasting here
    sum = (c_0_2_true.T.unsqueeze(1) * c_0_1_true.T.unsqueeze(2) * dp.data_dict[0]["density"].view(-1, 1, 1)).sum()

    print(sum.squeeze(), kernel_sum.squeeze())
    assert torch.allclose(
        sum.view(-1), kernel_sum.view(-1), atol=1e-8
    ), "Kernel size reduction failed"


@pytest.mark.parametrize(
    "n1, n2, m1, m2, L, grid_type",
    [
        (11, 10, 5, 7, 0.9, "flat"),
        (8, 8, 13, 8, 3.5, "tensor"),
        (12, 11, 9, 9, 2.0, "tuple"),
    ],
)  # noqa: E501
def test_kernel_with_different_grid_random_density_prod_false(
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

    # kernal size reduction 
    kernel_sum = kernel_size(dp, epsilon, prod=False)

    alpha = alpha_reduction(dp, 0, 1, epsilon=epsilon, prod=False)

    # Torch version for comparison
    c_0_1_true = (
        torch.exp(
            (
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
        / np.prod(dp.data_dict[1]["f"].shape)
    )

    # Torch version for comparison
    c_0_2_true = (
        torch.exp(
            (
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
        / np.prod(dp.data_dict[2]["f"].shape)
    )

    # Need some sort of 3D broadcasting here
    sum = (c_0_2_true.T.unsqueeze(1) * c_0_1_true.T.unsqueeze(2) / np.prod(dp.data_dict[0]["f"].shape)).sum()

    print(sum.squeeze(), kernel_sum.squeeze())
    assert torch.allclose(
        sum.view(-1), kernel_sum.view(-1), atol=1e-8
    ), "Kernel size reduction failed"

if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main([__file__]))
