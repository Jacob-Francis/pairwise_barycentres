import pytest
import torch
from mmuot import alpha_reduction, generate_mmuotdataprocessor_star_graph
import numpy as np
import networkx as nx


# --------------------------------------------------
#                   Test recusion
# --------------------------------------------------


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
def test_alpha_reduction_with_same_grid_uniform_density(n1, n2, L, grid_type):

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

    alpha = alpha_reduction(dp, 0, 1, epsilon=epsilon, prod=False)

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

    assert torch.allclose(
        alpha.view(-1), a_0_1_true.view(-1), atol=1e-8
    ), "Alpha reduction recursion failed"

    alpha = alpha_reduction(dp, 0, 2, epsilon=epsilon, prod=False)

    # Torch version for comparison
    a_0_3_true = torch.exp(
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
    ).sum(0) / np.prod(
        dp.data_dict[2]["f"].shape
    )  # times none since 3 is a leaf node

    assert torch.allclose(
        alpha.view(-1), a_0_3_true.view(-1), atol=1e-8
    ), "Alpha reduction recursion failed"

    alpha = alpha_reduction(dp, 2, 0, epsilon=epsilon, prod=False)

    # Torch version for comparison
    a_2_0_true = (
        (
            torch.exp(
                (
                    dp.data_dict[0]["f"].view(-1, 1)
                    - torch.cdist(
                        Y,
                        Y,
                    )
                    ** 2
                    * dp.graph[0][2]["weight"]
                    / 2
                )
                / epsilon
            )
            * a_0_1_true.view(-1, 1)
        )
        .sum(0)
        .view(
            -1,
        )
    ) / np.prod(dp.data_dict[0]["f"].shape)

    assert torch.allclose(
        alpha.view(-1), a_2_0_true.view(-1), atol=1e-5
    ), "Alpha reduction recursion failed"


@pytest.mark.parametrize(
    "n1, n2, m1, m2, L, grid_type",
    [
        (11, 10, 5, 7, 0.9, "flat"),
        (8, 8, 13, 8, 3.5, "tensor"),
        (12, 11, 9, 9, 2.0, "tuple"),
    ],
)  # noqa: E501
def test_alpha_reduction_with_different_grid_uniform_density(
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
        data.append([None, Y])  # central grid
        for m in range(members):  # member grids
            X = torch.cartesian_prod(
                torch.linspace(0, L, n1 + np.random.randint(-members, members)),
                torch.linspace(0, L, n2 + np.random.randint(-members, members)),
            ).type(torch.DoubleTensor)
            data.append([None, X])  # uniform density, grid will equal everywhere

    elif grid_type == "tensor":
        data = []
        Y = torch.stack(
            torch.meshgrid(
                torch.linspace(0, L, m1), torch.linspace(0, L, m2), indexing="ij"
            ),
            dim=-1,
        ).type(torch.DoubleTensor)
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
            data.append([None, X])

    elif grid_type == "tuple":
        toggle = False
        data = []
        Y = (torch.linspace(0, L, m1), torch.linspace(0, L, m2))
        data.append([None, Y])  # central grid
        for m in range(members):
            X = (
                torch.linspace(0, L, n1 + np.random.randint(-members, members)),
                torch.linspace(0, L, n2 + np.random.randint(-members, members)),
            )
            data.append([None, X])

    # generate the barycentre dataprocessor class which will store all objects
    # of interest. It will also create the correct graph, and given no density of graphs
    # will create uniform densities on the grids
    dp = generate_mmuotdataprocessor_star_graph(data, grid=None, clear_grid=False)
    epsilon = dp._torch_numpy_process(
        max(L / np.sqrt(n1 * n2), L / np.sqrt(m1 * m2))
    ).view(-1, 1)

    alpha = alpha_reduction(dp, 0, 1, epsilon=epsilon, prod=False)

    # Torch version for comparison
    a_0_1_true = torch.exp(
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
    ).sum(0) / np.prod(
        dp.data_dict[1]["f"].shape
    )  # times none since 2 is a leaf node

    assert torch.allclose(
        alpha.view(-1), a_0_1_true.view(-1), atol=1e-8
    ), "Alpha reduction recursion failed"

    alpha = alpha_reduction(dp, 0, 2, epsilon=epsilon, prod=False)

    # Torch version for comparison
    a_0_3_true = torch.exp(
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
    ).sum(0) / np.prod(
        dp.data_dict[2]["f"].shape
    )  # times none since 3 is a leaf node

    assert torch.allclose(
        alpha.view(-1), a_0_3_true.view(-1), atol=1e-8
    ), "Alpha reduction recursion failed"

    alpha = alpha_reduction(dp, 2, 0, epsilon=epsilon, prod=False)

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
        )
        .sum(0)
        .view(
            -1,
        )
    ) / np.prod(dp.data_dict[0]["f"].shape)

    assert torch.allclose(
        alpha.view(-1), a_2_0_true.view(-1), atol=1e-5
    ), "Alpha reduction recursion failed"


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
    a_0_3_true = (
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
        alpha.view(-1), a_0_3_true.view(-1), atol=1e-8
    ), "Alpha reduction recursion failed"

    alpha = alpha_reduction(dp, 2, 0, epsilon=epsilon, prod=True)

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

    assert torch.allclose(
        alpha.view(-1), a_2_0_true.view(-1), atol=1e-5
    ), "Alpha reduction recursion failed"


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main([__file__]))
