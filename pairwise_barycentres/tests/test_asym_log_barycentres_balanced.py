
########## Same as test_asym_log_barycenter_balanced_margianls basically

# import numpy as np
# from scipy.spatial.distance import cdist
# import torch
# import pytest
# from pwbarycentres import (
#     asymmetric_sinkhorn_log_algorithm,
#     generate_barycentredataprocessor,
# )
# import networkx as nx

# torch.set_printoptions(precision=8)

# # avoid scipy dependence
# def numpy_sqdist_matrix(X, Y):
#     """
#     Return matrix D of squared distances between rows of X (n x d) and Y (m x d),
#     shape (n, m).
#     """
#     # Using (x - y)^2 = ||x||^2 + ||y||^2 - 2 x.y
#     X2 = np.sum(X**2, axis=1)[:, None]  # (n,1)
#     Y2 = np.sum(Y**2, axis=1)[None, :]  # (1,m)
#     XY = X @ Y.T  # (n,m)
#     D = X2 + Y2 - 2 * XY
#     return D


# # --------------------------------------------------------
# # Testing barycentre and other grids which are the same
# # --------------------------------------------------------
# @pytest.mark.parametrize(
#     "n1, n2, members, L, grid_type",
#     [
#         (8, 8, 6, 3.5, "tensor"),
#         (9, 8, 7, 3.5, "tensor"),
#         (8, 9, 8, 3.5, "tensor"),
#         (11, 11, 2, 0.9, "flat"),
#         (11, 10, 3, 0.9, "flat"),
#         (11, 12, 2, 0.9, "flat"),
#         (12, 12, 3, 6.0, "tuple"),
#         (12, 13, 3, 6.0, "tuple"),
#         (12, 11, 3, 6.0, "tuple"),
#     ],
# )  # noqa: E501
# def test_asym_log_bary_with_same_grid_uniform_density_without_debiasing(
#     n1, n2, members, L, grid_type
# ):
#     dx = L / n1
#     dy = L / n2
#     if grid_type == "flat":
#         X = torch.cartesian_prod(
#             torch.linspace(dx/2, L - dx/2, n1), torch.linspace(dy/2, L - dy/2, n2)
#         ).type(torch.DoubleTensor)
#     elif grid_type == "tensor":
#         X = torch.stack(
#             torch.meshgrid(
#                 torch.linspace(dx/2, L - dx/2, n1), torch.linspace(dy/2, L - dy/2, n2), indexing="ij"
#             ),
#             dim=-1,
#         ).type(torch.DoubleTensor)
#     elif grid_type == "tuple":
#         X = (torch.linspace(dx/2, L - dx/2, n1), torch.linspace(dy/2, L - dy/2, n2))

#     data = []

#     for _ in range(members):
#         data.append([None, None])  # uniform density, grid will equal everywhere

#     cell_areas = L**2 / (n1 * n2)
#     # generate the barycentre dataprocessor class which will store all objects
#     # of interest. It will also create the correct graph, and given no density of graphs
#     # will create uniform densities on the grids
#     data_processor = generate_barycentredataprocessor(data, barycentre_grid=X, grid=X, potentials='f')
    
#     # for nodes in data_processor.graph.nodes():
#     #     print('areas', cell_areas, data_processor.data_dict[nodes]["cell_areas"])
#     #     print('true:', data_processor.data_dict[nodes]["density"].sum().item(), (data_processor.data_dict[nodes]["density"]*cell_areas).sum().item())
#     #     # print('marginal:', marginals[nodes]["marginal"].sum().item(), (marginals[nodes]["marginal"]*cell_areas).sum().item(), marginals[nodes]["error"])

        
#     # Assert that the orginal structure is correct
#     for edges in data_processor.graph.edges():
#         assert (
#             np.abs(data_processor.data_dict[edges[0]]["density"].sum().item()*cell_areas - 1.0)
#             < 1e-5
#         ), (data_processor.data_dict[edges[0]]["density"].sum().item()*cell_areas)
#         assert (
#             np.abs(data_processor.data_dict[edges[1]]["density"].sum().item()*cell_areas - 1.0)
#             < 1e-5
#         ), (data_processor.data_dict[edges[1]]["density"].sum().item()*cell_areas)

#     # run asymmetric sinkhorn algorithm
#     data_processor, barycentre, potential_error_list, barycentre_error_list = (
#         asymmetric_sinkhorn_log_algorithm(
#             data_processor,
#             epsilon=1 / np.sqrt(n1 * n2),
#             rho=1.0,
#             aprox="balanced",
#             max_iterates=500,
#             tol=1e-12,
#             epsilon_annealing=False,
#             debiasing=False,
#         )
#     )

#     assert barycentre_error_list[-1] < 1e-7  # less than tolerance
#     assert np.abs(barycentre.sum().item()*cell_areas - 1.0) < 1e-5, barycentre.sum().item()*cell_areas

#     # assert after structure is correct
#     for edges in data_processor.graph.edges():
#         assert (
#             np.abs(data_processor.data_dict[edges[0]]["density"].sum().item()*cell_areas - 1.0)
#             < 1e-5
#         ), (data_processor.data_dict[edges[0]]["density"].sum().item()*cell_areas)
#         assert (
#             np.abs(data_processor.data_dict[edges[1]]["density"].sum().item()*cell_areas - 1.0)
#             < 1e-5
#         ), (data_processor.data_dict[edges[1]]["density"].sum().item()*cell_areas)

#    # the biased version doesn't match well with the uniform test 
#    # its contracts at the boundaries so can't test easily. 

# @pytest.mark.parametrize(
#     "n1, n2, members, L, grid_type",
#     [
#         (8, 8, 6, 3.5, "tensor"),
#         (9, 8, 7, 3.5, "tensor"),
#         (8, 9, 8, 3.5, "tensor"),
#         (11, 11, 2, 1.0, "flat"),
#         (8,7, 3, 1.0, "flat"),
#         (11, 12, 2, 1.0, "flat"),
#         (12, 12, 3, 6.0, "tuple"),
#         (12, 13, 3, 6.0, "tuple"),
#         (12, 11, 3, 6.0, "tuple"),
#     ],
# )  # noqa: E501
# def test_asym_log_bary_with_same_grid_uniform_density_with_debiasing(
#     n1, n2, members, L, grid_type
# ):

#     dx = L / n1
#     dy = L / n2
#     if grid_type == "flat":
#         X = torch.cartesian_prod(
#             torch.linspace(dx/2, L-dx/2, n1), torch.linspace(dy/2, L-dy/2, n2)
#         ).type(torch.DoubleTensor)
#     elif grid_type == "tensor":
#         X = torch.stack(
#             torch.meshgrid(
#                 torch.linspace(dx/2, L-dx/2, n1), torch.linspace(dy/2, L-dy/2, n2), indexing="ij"
#             ),
#             dim=-1,
#         ).type(torch.DoubleTensor)
#     elif grid_type == "tuple":
#         X = (torch.linspace(dx/2, L-dx/2, n1), torch.linspace(dy/2, L-dy/2, n2))

#     data = []

#     for m in range(members):
#         data.append([None, None])  # uniform density, grid will equal everywhere

#     cell_areas = L**2 / (n1 * n2)
#     # generate the barycentre dataprocessor class which will store all objects
#     # of interest. It will also create the correct graph, and given no density of graphs
#     # will create uniform densities on the grids
#     data_processor = generate_barycentredataprocessor(data, barycentre_grid=X, grid=X, potentials='f')

#     # Assert that the orginal structure is correct
#     for edges in data_processor.graph.edges():
#         assert (
#             np.abs(data_processor.data_dict[edges[0]]["density"].sum().item()*cell_areas - 1.0)
#             < 1e-5
#         ), (data_processor.data_dict[edges[0]]["density"].sum().item()*cell_areas)
#         assert (
#             np.abs(data_processor.data_dict[edges[1]]["density"].sum().item()*cell_areas - 1.0)
#             < 1e-5
#         ), (data_processor.data_dict[edges[1]]["density"].sum().item()*cell_areas)

#     # run asymmetric sinkhorn algorithm
#     data_processor, barycentre, potential_error_list, barycentre_error_list = (
#         asymmetric_sinkhorn_log_algorithm(
#             data_processor,
#             epsilon=1 / np.sqrt(n1 * n2),
#             rho=1.0,
#             aprox="balanced",
#             max_iterates=1000,
#             tol=1e-12,  # relax tolerance because it was not converging very fast for this debiased setting?
#             epsilon_annealing=False,
#             debiasing=True,
#         )
#     )

#     # to speed up tests i don't necessarily give enough its
#     assert barycentre_error_list[-1] < 1e-4, f"its {len(barycentre_error_list)}, err {barycentre_error_list[-1]}"  # less than tolerance
#     assert np.abs(barycentre.sum().item()*cell_areas - 1.0) < 1e-3

#     for edges in data_processor.graph.edges():
#         assert (
#             np.abs(data_processor.data_dict[edges[0]]["density"].sum().item()*cell_areas - 1.0)
#             < 1e-3
#         ), (data_processor.data_dict[edges[0]]["density"].sum().item()*cell_areas)
#         assert (
#             np.abs(data_processor.data_dict[edges[1]]["density"].sum().item()*cell_areas - 1.0)
#             < 1e-3
#         ), (data_processor.data_dict[edges[1]]["density"].sum().item()*cell_areas)

#     # Since using a uniform density the barycentre should also be uniform
#     # Because of debiasing we can make the tolerance tighter
#     print(barycentre.view(-1)[:10])
#     print((torch.ones_like(barycentre) / barycentre.numel() / cell_areas).view(-1)[:10])
#     assert torch.allclose(
#         barycentre, torch.ones_like(barycentre) / barycentre.numel() / cell_areas, atol=1e-1
#     ), torch.norm(barycentre - torch.ones_like(barycentre) / barycentre.numel() / cell_areas, p=float("inf")).item()


# # --------------------------------------------------------
# # Testing barycentre and other grids which are different
# # --------------------------------------------------------
# @pytest.mark.parametrize(
#     "n1, n2, members, m1, m2, L, grid_type",
#     [
#         (11, 10, 3, 5, 7, 0.9, "flat"),
#         (8, 8, 4, 13, 8, 3.5, "tensor"),
#         (12, 11, 3, 9, 9, 2.0, "tuple"),
#     ],
# )  # noqa: E501
# def test_asym_log_bary_with_different_grid_uniform_density_without_debiasing(
#     n1, n2, members, m1, m2, L, grid_type
# ):

#     if grid_type == "flat":
#         X = torch.cartesian_prod(
#             torch.linspace(0, L, n1), torch.linspace(0, L, n2)
#         ).type(torch.DoubleTensor)
#         Y = torch.cartesian_prod(
#             torch.linspace(0, L, m1), torch.linspace(0, L, m2)
#         ).type(torch.DoubleTensor)
#     elif grid_type == "tensor":
#         X = torch.stack(
#             torch.meshgrid(
#                 torch.linspace(0, L, n1), torch.linspace(0, L, n2), indexing="ij"
#             ),
#             dim=-1,
#         ).type(torch.DoubleTensor)
#         Y = torch.stack(
#             torch.meshgrid(
#                 torch.linspace(0, L, m1), torch.linspace(0, L, m2), indexing="ij"
#             ),
#             dim=-1,
#         ).type(torch.DoubleTensor)
#     elif grid_type == "tuple":
#         X = (torch.linspace(0, L, n1), torch.linspace(0, L, n2))
#         Y = (torch.linspace(0, L, m1), torch.linspace(0, L, m2))

#     data = []

#     for m in range(members):
#         data.append([None, X])  # uniform density, grid will equal everywhere

#     # generate the barycentre dataprocessor class which will store all objects
#     # of interest. It will also create the correct graph, and given no density of graphs
#     # will create uniform densities on the grids
#     data_processor = generate_barycentredataprocessor(data, barycentre_grid=Y, potentials='f')

#     # run asymmetric sinkhorn algorithm
#     data_processor, barycentre, potential_error_list, barycentre_error_list = (
#         asymmetric_sinkhorn_log_algorithm(
#             data_processor,
#             epsilon=max(1 / np.sqrt(n1 * n2), 1 / np.sqrt(m1 * m2)),
#             rho=1.0,
#             aprox="balanced",
#             max_iterates=500,
#             tol=1e-7,
#             epsilon_annealing=False,
#             debiasing=False,
#         )
#     )

#     assert barycentre_error_list[-1] < 1e-7, len(barycentre_error_list)  # less than tolerance

#     for edges in data_processor.graph.edges():
#         assert np.isclose(
#             (data_processor.data_dict[edges[0]]["density"]*data_processor.data_dict[edges[0]]["cell_areas"]).sum().item(), 1.0
#         )
#         assert np.isclose(
#             (data_processor.data_dict[edges[1]]["density"]*data_processor.data_dict[edges[1]]["cell_areas"]).sum().item(), 1.0
#         )

#     # The uniform test is too strict when the grids differ


# @pytest.mark.parametrize(
#     "n1, n2, members, m1, m2, L, grid_type",
#     [
#         (11, 10, 3, 5, 7, 0.9, "flat"),
#         (8, 8, 4, 13, 8, 3.5, "tensor"),
#         (12, 11, 3, 9, 9, 2.0, "tuple"),
#     ],
# )  # noqa: E501
# def test_asym_log_bary_with_different_grid_uniform_density_with_debiasing(
#     n1, n2, members, m1, m2, L, grid_type
# ):

#     if grid_type == "flat":
#         X = torch.cartesian_prod(
#             torch.linspace(0, L, n1), torch.linspace(0, L, n2)
#         ).type(torch.DoubleTensor)
#         Y = torch.cartesian_prod(
#             torch.linspace(0, L, m1), torch.linspace(0, L, m2)
#         ).type(torch.DoubleTensor)
#     elif grid_type == "tensor":
#         X = torch.stack(
#             torch.meshgrid(
#                 torch.linspace(0, L, n1), torch.linspace(0, L, n2), indexing="ij"
#             ),
#             dim=-1,
#         ).type(torch.DoubleTensor)
#         Y = torch.stack(
#             torch.meshgrid(
#                 torch.linspace(0, L, m1), torch.linspace(0, L, m2), indexing="ij"
#             ),
#             dim=-1,
#         ).type(torch.DoubleTensor)
#     elif grid_type == "tuple":
#         X = (torch.linspace(0, L, n1), torch.linspace(0, L, n2))
#         Y = (torch.linspace(0, L, m1), torch.linspace(0, L, m2))

#     data = []

#     for m in range(members):
#         data.append([None, X])  # uniform density, grid will equal everywhere

#     # generate the barycentre dataprocessor class which will store all objects
#     # of interest. It will also create the correct graph, and given no density of graphs
#     # will create uniform densities on the grids
#     data_processor = generate_barycentredataprocessor(data, barycentre_grid=Y, potentials='f')

#     # run asymmetric sinkhorn algorithm
#     data_processor, barycentre, potential_error_list, barycentre_error_list = (
#         asymmetric_sinkhorn_log_algorithm(
#             data_processor,
#             epsilon=max(1 / np.sqrt(n1 * n2), 1 / np.sqrt(m1 * m2)),
#             rho=1.0,
#             aprox="balanced",
#             max_iterates=2000,
#             tol=1e-12,
#             epsilon_annealing=False,
#             debiasing=True,
#         )
#     )

#     assert barycentre_error_list[-1] < 1e-4  # less than tolerance

#     for edges in data_processor.graph.edges():
#         assert np.isclose(
#             (data_processor.data_dict[edges[0]]["density"]*data_processor.data_dict[edges[0]]["cell_areas"]).sum().item(), 1.0
#         )
#         assert np.isclose(
#             (data_processor.data_dict[edges[1]]["density"]*data_processor.data_dict[edges[1]]["cell_areas"]).sum().item(), 1.0
#         )


# # -------------------------------------------------------------------------------------------
# # Testing all on different grids (though still same tpye for now)
# # -------------------------------------------------------------------------------------------
# @pytest.mark.parametrize(
#     "n1, n2, members, m1, m2, L, grid_type",
#     [
#         (11, 10, 3, 5, 7, 0.9, "flat"),
#         (8, 8, 4, 13, 8, 3.5, "tensor"),
#         (12, 11, 3, 9, 9, 2.0, "tuple"),
#     ],
# )  # noqa: E501
# def test_asym_log_bary_with_all_different_grids_with_debiasing(
#     n1, n2, members, m1, m2, L, grid_type
# ):

#     np.random.seed(n1 * n2 * members * m1 * m2)

#     if grid_type == "flat":
#         data = []
#         for m in range(members):
#             X = torch.cartesian_prod(
#                 torch.linspace(0, L, n1 + np.random.randint(-members, members)),
#                 torch.linspace(0, L, n2 + np.random.randint(-members, members)),
#             ).type(torch.DoubleTensor)
#             data.append([None, X])  # uniform density, grid will equal everywhere

#         Y = torch.cartesian_prod(
#             torch.linspace(0, L, m1), torch.linspace(0, L, m2)
#         ).type(torch.DoubleTensor)
#     elif grid_type == "tensor":
#         data = []
#         for m in range(members):
#             X = torch.stack(
#                 torch.meshgrid(
#                     torch.linspace(0, L, n1 + np.random.randint(-members, members)),
#                     torch.linspace(0, L, n2 + np.random.randint(-members, members)),
#                     indexing="ij",
#                 ),
#                 dim=-1,
#             ).type(torch.DoubleTensor)
#             data.append([None, X])
#         Y = torch.stack(
#             torch.meshgrid(
#                 torch.linspace(0, L, m1), torch.linspace(0, L, m2), indexing="ij"
#             ),
#             dim=-1,
#         ).type(torch.DoubleTensor)
#     elif grid_type == "tuple":
#         data = []
#         for m in range(members):
#             X = (
#                 torch.linspace(0, L, n1 + np.random.randint(-members, members)),
#                 torch.linspace(0, L, n2 + np.random.randint(-members, members)),
#             )
#             data.append([None, X])
#         Y = (torch.linspace(0, L, m1), torch.linspace(0, L, m2))

#     # generate the barycentre dataprocessor class which will store all objects
#     # of interest. It will also create the correct graph, and given no density of graphs
#     # will create uniform densities on the grids
#     data_processor = generate_barycentredataprocessor(data, barycentre_grid=Y, potentials='f')

#     # run asymmetric sinkhorn algorithm
#     data_processor, barycentre, potential_error_list, barycentre_error_list = (
#         asymmetric_sinkhorn_log_algorithm(
#             data_processor,
#             epsilon=max(1 / np.sqrt(n1 * n2), 1 / np.sqrt(m1 * m2)),
#             rho=1.0,
#             aprox="balanced",
#             max_iterates=1000,
#             tol=1e-12,
#             epsilon_annealing=False,
#             debiasing=True,
#             verbose=True,
#         )
#     )

#     assert barycentre_error_list[-1] < 1e-3  # less than tolerance

#     for edges in data_processor.graph.edges():
#         assert np.isclose(
#             (data_processor.data_dict[edges[0]]["density"] * data_processor.data_dict[edges[0]]["cell_areas"]).sum().item(), 1.0, atol=1e-3
#         )
#         assert np.isclose(
#             (data_processor.data_dict[edges[1]]["density"] * data_processor.data_dict[edges[1]]["cell_areas"]).sum().item(), 1.0, atol=1e-3
#         )


# @pytest.mark.parametrize(
#     "n1, n2, members, m1, m2, L, grid_type",
#     [
#         (11, 10, 2, 5, 7, 0.9, "flat"),
#         (8, 8, 4, 13, 8, 3.5, "tensor"),
#         (12, 11, 3, 9, 9, 2.0, "tuple"),
#     ],
# )  # noqa: E501
# def test_asym_log_bary_with_all_different_grids_without_debiasing(
#     n1, n2, members, m1, m2, L, grid_type
# ):

#     np.random.seed(n1 * n2 * members * m1 * m2)

#     if grid_type == "flat":
#         data = []
#         for m in range(members):
#             X = torch.cartesian_prod(
#                 torch.linspace(0, L, n1 + np.random.randint(-members, members)),
#                 torch.linspace(0, L, n2 + np.random.randint(-members, members)),
#             ).type(torch.DoubleTensor)
#             data.append([None, X])  # uniform density, grid will equal everywhere

#         Y = torch.cartesian_prod(
#             torch.linspace(0, L, m1), torch.linspace(0, L, m2)
#         ).type(torch.DoubleTensor)
#     elif grid_type == "tensor":
#         data = []
#         for m in range(members):
#             X = torch.stack(
#                 torch.meshgrid(
#                     torch.linspace(0, L, n1 + np.random.randint(-members, members)),
#                     torch.linspace(0, L, n2 + np.random.randint(-members, members)),
#                     indexing="ij",
#                 ),
#                 dim=-1,
#             ).type(torch.DoubleTensor)
#             data.append([None, X])
#         Y = torch.stack(
#             torch.meshgrid(
#                 torch.linspace(0, L, m1), torch.linspace(0, L, m2), indexing="ij"
#             ),
#             dim=-1,
#         ).type(torch.DoubleTensor)
#     elif grid_type == "tuple":
#         data = []
#         for m in range(members):
#             X = (
#                 torch.linspace(0, L, n1 + np.random.randint(-members, members)),
#                 torch.linspace(0, L, n2 + np.random.randint(-members, members)),
#             )
#             data.append([None, X])
#         Y = (torch.linspace(0, L, m1), torch.linspace(0, L, m2))

#     # generate the barycentre dataprocessor class which will store all objects
#     # of interest. It will also create the correct graph, and given no density of graphs
#     # will create uniform densities on the grids
#     data_processor = generate_barycentredataprocessor(data, barycentre_grid=Y, potentials='f')

#     # run asymmetric sinkhorn algorithm
#     data_processor, barycentre, potential_error_list, barycentre_error_list = (
#         asymmetric_sinkhorn_log_algorithm(
#             data_processor,
#             epsilon=max(1 / np.sqrt(n1 * n2), 1 / np.sqrt(m1 * m2)),
#             rho=1.0,
#             aprox="balanced",
#             max_iterates=1000,
#             tol=1e-12,
#             epsilon_annealing=False,
#             debiasing=False,
#             verbose=True,
#         )
#     )

#     assert barycentre_error_list[-1] < 1e-3  # less than tolerance

#     for edges in data_processor.graph.edges():
#         assert np.isclose(
#             (data_processor.data_dict[edges[0]]["density"] * data_processor.data_dict[edges[0]]["cell_areas"]).sum().item(), 1.0, atol=1e-3
#         )
#         assert np.isclose(
#             (data_processor.data_dict[edges[1]]["density"] * data_processor.data_dict[edges[1]]["cell_areas"]).sum().item(), 1.0, atol=1e-3
#         )


# if __name__ == "__main__":
#     import sys

#     pytest.main(sys.argv)
