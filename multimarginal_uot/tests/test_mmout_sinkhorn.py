import pytest
import torch
from mmuot import alpha_reduction, generate_mmuotdataprocessor_star_graph, sinkhorn_update
import numpy as np
import networkx as nx

@pytest.mark.parametrize(
    "n1, n2, L, grid_type",
    [   (11, 11, 0.9, "flat"),
        (11, 10,  0.9, "flat"),
        (11, 12,  0.9, "flat"),
        (9, 8, 3.5, "tensor"),
        (8, 9, 3.5, "tensor"),
        (8, 8, 3.5, "tensor"),
        (12, 12,  6.0, "tuple"),
        (12, 13,  6.0, "tuple"),
        (12, 11,  6.0, "tuple"),

    ],
)  # noqa: E501
def test_sinkhorn_reduction_with_same_grid_uniform_density_uniform_measure_root_node(n1, n2, L, grid_type):

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
    Y = torch.cartesian_prod(
            torch.linspace(0, L, n1), torch.linspace(0, L, n2)
        ).type(torch.DoubleTensor)

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
    ).sum(
        0
    ) / np.prod(dp.data_dict[1]["f"].shape) # times none since 2 is a leaf node
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
    ).sum(
        0
    ) / np.prod(dp.data_dict[2]["f"].shape) 

    f_new = (
        dp.data_dict[0]["f"].view(-1)
        + epsilon * torch.log(dp.data_dict[0]["density"].view(-1))
        - epsilon * torch.log(a_0_1_true.view(-1) * a_0_2_true.view(-1))
    )

    # Check against what is should be
    for p_j, j in reversed(list(nx.dfs_tree(dp.graph, source=0).edges)):
        dp.data_dict[(p_j, j)]['alpha'] = alpha_reduction(dp, p_j,j, epsilon=epsilon, prod=False)

    f, err = sinkhorn_update(dp, 0, epsilon, rho=1.0, aprox='balanced', prod=False)

    assert torch.allclose(
        f_new.view(-1), f.view(-1), atol=1e-8
    ), "Sinkhorn update failed"

# --------------------------------------------------
@pytest.mark.parametrize(
    "n1, n2, L, grid_type",
    [   (11, 11, 0.9, "flat"),
        (11, 10,  0.9, "flat"),
        (11, 12,  0.9, "flat"),
        (9, 8, 3.5, "tensor"),
        (8, 9, 3.5, "tensor"),
        (8, 8, 3.5, "tensor"),
        (12, 12,  6.0, "tuple"),
        (12, 13,  6.0, "tuple"),
        (12, 11,  6.0, "tuple"),

    ],
)  # noqa: E501
def test_sinkhorn_reduction_with_same_grid_uniform_density_uniform_measure_leaf_node(n1, n2, L, grid_type):

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
    Y = torch.cartesian_prod(
            torch.linspace(0, L, n1), torch.linspace(0, L, n2)
        ).type(torch.DoubleTensor)

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
    ).sum(
        0
    ) / np.prod(dp.data_dict[2]["f"].shape) 

    a_1_0_true = (torch.exp(
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
    )*a_0_2_true).sum(
        0
    ) / np.prod(dp.data_dict[0]["f"].shape) # times none since 2 is a leaf node

    f_new = (
        dp.data_dict[1]["f"].view(-1)
        + epsilon * torch.log(dp.data_dict[1]["density"].view(-1))
        - epsilon * torch.log(a_1_0_true.view(-1))
    )

    # Check against what is should be
    for p_j, j in reversed(list(nx.dfs_tree(dp.graph, source=0).edges)):
        dp.data_dict[(p_j, j)]['alpha'] = alpha_reduction(dp, p_j,j, epsilon=epsilon, prod=False)
    
    # Check against what is should be
    for p_j, j in list(nx.dfs_tree(dp.graph, source=0).edges):
        dp.data_dict[(j, p_j)]['alpha'] = alpha_reduction(dp, j,p_j, epsilon=epsilon, prod=False)

    f, err = sinkhorn_update(dp, 1, epsilon, rho=1.0, aprox='balanced', prod=False)

    assert torch.allclose(
        f_new.view(-1), f.view(-1), atol=1e-8
    ), "Sinkhorn update failed"


# @pytest.mark.parametrize("num_1,num_2", [(10, 11), (20, 50), (5, 17)])
# def test_sinkhorn_update_leaf_node(num_1, num_2):
#     dimension = 2
#     epsilon = torch.Tensor([0.5])

#     G = build_star_graph(2, weights=None, plot_graph=False)
#     grid_1 = torch.cartesian_prod(
#         torch.linspace(0, 1, num_1),
#         torch.Tensor([1.0]),
#     ).view(num_1, dimension)
#     grid_2 = torch.cartesian_prod(
#         torch.linspace(0, 1, num_2),
#         torch.Tensor([1.0]),
#     ).view(num_2, dimension)

#     # root node grid
#     num_i = max(num_1, num_2)
#     grid = torch.cartesian_prod(torch.linspace(0, 1, num_i), torch.Tensor([1.0])).view(
#         num_i, dimension
#     )

#     # transport data to float64
#     data = [
#         [torch.ones(num_i) / num_i, grid],
#         [torch.ones(num_1) / num_1, grid_1],
#         [torch.ones(num_2) / num_2, grid_2],
#     ]
#     for k in range(3):
#         data[k][0] = data[k][0].to(torch.float64)
#         data[k][1] = data[k][1].to(torch.float64)

#     # Initialise graph dictionary
#     graph_dictionary = initialise_graph_dictionary(G, data)

#     # To process to node 3, you first need to update the root node

#     # Reverse pass, intialising the alpha values
#     for p_j, j in reversed(list(nx.dfs_tree(G, source=1).edges)):
#         alpha_reduction(p_j, j, graph_dictionary, epsilon)

#     # ##############################################################
#     # Before calauting the leaf node update, we need to calculate the true values since these functions work in place
#     # ##############################################################

#     node = 1
#     # Calcuate projection
#     a_1_2_true = torch.exp(
#         (
#             graph_dictionary["nodes"][2]["f"].view(-1, 1)
#             - torch.cdist(
#                 graph_dictionary["nodes"][2]["grid"],
#                 graph_dictionary["nodes"][1]["grid"],
#             )
#             ** 2
#             * G[1][2]["weight"]
#             / 2
#         )
#         / epsilon
#     ).sum(
#         0
#     )  # times none since 2 is a leaf node
#     # Torch version for comparison
#     a_1_3_true = torch.exp(
#         (
#             graph_dictionary["nodes"][3]["f"].view(-1, 1)
#             - torch.cdist(
#                 graph_dictionary["nodes"][3]["grid"],
#                 graph_dictionary["nodes"][1]["grid"],
#             )
#             ** 2
#             * G[1][3]["weight"]
#             / 2
#         )
#         / epsilon
#     ).sum(
#         0
#     )  # times none since 2 is a leaf node

#     f_new = (
#         graph_dictionary["nodes"][node]["f"]
#         + epsilon * torch.log(graph_dictionary["nodes"][node]["data"])
#         - epsilon * torch.log(a_1_2_true.view(-1) * a_1_3_true.view(-1))
#     )

#     # First root node update
#     _ = sinkhorn_update(1, graph_dictionary, epsilon)

#     assert torch.allclose(
#         f_new, graph_dictionary["nodes"][node]["f"], atol=1e-8
#     ), "Sinkhorn update failed"

#     # ##############################################################
#     # Now we check node 3
#     # ##############################################################
#     node = 3

#     # Torch version for comparison
#     a_3_1_true = (
#         (
#             torch.exp(
#                 (
#                     graph_dictionary["nodes"][1]["f"].view(-1, 1)
#                     - torch.cdist(
#                         graph_dictionary["nodes"][1]["grid"],
#                         graph_dictionary["nodes"][3]["grid"],
#                     )
#                     ** 2
#                     * G[1][3]["weight"]
#                     / 2
#                 )
#                 / epsilon
#             )
#             * a_1_2_true.view(-1, 1)
#         )
#         .sum(0)
#         .view(
#             -1,
#         )
#     )

#     a = alpha_reduction(node, 1, graph_dictionary, epsilon=epsilon, output=True)
#     assert torch.allclose(a, a_3_1_true, atol=1e-8)

#     f_new = (
#         graph_dictionary["nodes"][node]["f"]
#         + epsilon * torch.log(graph_dictionary["nodes"][node]["data"])
#         - epsilon * torch.log(a_3_1_true.view(-1))
#     )

#     err = sinkhorn_update(node, graph_dictionary, epsilon)

#     assert torch.allclose(
#         f_new, graph_dictionary["nodes"][node]["f"], atol=1e-8
#     ), "Sinkhorn update failed"



if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main([__file__]))

