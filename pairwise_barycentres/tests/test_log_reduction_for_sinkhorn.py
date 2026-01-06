from pwbarycentres import _log_reduction_for_sinkhorn


# def _log_reduction_for_sinkhorn(dp, k, edge, epsilon, debiasing=True):
#     """

#     :param dp: Description
#     :param k: Description
#     :param edge: Description
#     :param epsilon: Description
#     """

#     assert k in edge

#     # Perform reduction to node k across edge with the kernel Kd or K.
#     bary_node = edge[0]
#     data_node = edge[1]

#     if debiasing:
#         if "debiased_potential" in dp.data_dict[bary_node]:
#             b = (
#                 torch.exp(dp.data_dict[bary_node]["f"]/epsilon)
#                 * dp.data_dict[bary_node]["debiased_potential"]
#             )
#             a = dp.data_dict[data_node]["f"]
#         elif "debiased_potential" in dp.data_dict[data_node]:
#             raise Warning("No debiasing potentials should be attached to the data")
#         else:
#             raise Warning(
#                 "No debiasing potentials attached to either node, yet using debiasing"
#             )

#         if (
#             "debiased_potential" in dp.data_dict[bary_node]
#             and "debiased_potential" in dp.data_dict[data_node]
#         ):
#             raise Warning(
#                 "Both nodes have debiasing potentials attached, this is unexpected behaviour"
#             )
#     else:
#         a = dp.data_dict[data_node]["f"]
#         b = dp.data_dict[bary_node]["f"]

#     # Can I tensorise?
#     if "x1y1" in dp.data_dict[edge] and "x2y2" in dp.data_dict[edge]:
#         temp = _tensorised_sinkhorn_reduction(
#             a if k == data_node else b,
#             dp.data_dict[edge]["x1y1"],
#             dp.data_dict[edge]["x2y2"],
#             epsilon,
#         )

#         # testing for NaN/inf
#         if torch.any(torch.isnan(temp)) or torch.any(torch.isinf(temp)):
#             raise ValueError("Tensorised reduction NaN/inf detected", temp.sum().item(), k, edge)
#         else:
#             print("Tensorised reduction ok", temp.sum().item(), k, edge)

#         return temp 
#     # Otherwise PyKeOps
#     elif "grid" in dp.data_dict[edge[0]] and "grid" in dp.data_dict[edge[1]]:
#         temp = _flat_grid_log_sinkhorn_reduction(
#             dp.data_dict[k]["f"],
#             dp.data_dict[bary_node]["debiased_potential"] if (debiasing and k == bary_node) else torch.ones_like(dp.data_dict[k]["f"]),
#             dp.data_dict[k]["grid"],
#             dp.data_dict[edge[1] if edge[0] == k else edge[0]]["grid"],
#             epsilon,
#         )

#         # testing for NaN/inf
#         if torch.any(torch.isnan(temp)) or torch.any(torch.isinf(temp)):
            
#             print('sums', dp.data_dict[k]["f"].sum().item(),\
#                 (dp.data_dict[bary_node]["debiased_potential"] if (debiasing and k == bary_node) else torch.ones_like(dp.data_dict[k]["f"])).sum().item(),
#                 )
#             raise ValueError("Flat grid reduction NaN/inf detected", temp.sum().item(), k, edge)
#         else:
#             print("Flat grid reduction ok", temp.sum().item(), k, edge)
#         return temp


import numpy as np
from scipy.spatial.distance import cdist
import torch
import pytest
from pwbarycentres import (
    asymmetric_sinkhorn_log_algorithm,
    generate_barycentredataprocessor,
)
import networkx as nx

torch.set_printoptions(precision=8)

# avoid scipy dependence
def numpy_sqdist_matrix(X, Y):
    """
    Return matrix D of squared distances between rows of X (n x d) and Y (m x d),
    shape (n, m).
    """
    # Using (x - y)^2 = ||x||^2 + ||y||^2 - 2 x.y
    X2 = np.sum(X**2, axis=1)[:, None]  # (n,1)
    Y2 = np.sum(Y**2, axis=1)[None, :]  # (1,m)
    XY = X @ Y.T  # (n,m)
    D = X2 + Y2 - 2 * XY
    return D


@pytest.mark.parametrize(
    "n1, n2, members, m1, m2, L, grid_type",
    [        
        (8, 8, 3, 13, 8, 3.5, "tensor"),
        (11, 10, 3, 5, 7, 0.9, "flat"),
        (12, 11, 3, 9, 9, 2.0, "tuple"),
    ],
)  # noqa: E501
def test_log_reduction_update_debiasingFALSE_against_numpy(
    n1, n2, members, m1, m2, L, grid_type
):
    np.random.seed(12313*members + n1 + n2 + m1 + m2)

    # Generate grid
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
    data_processor = generate_barycentredataprocessor(data, barycentre_grid=Y, potentials='f')

    epsilon = 1 / max(np.sqrt(n1*n2), np.sqrt(m1*m2))
    epsilon = data_processor._torch_numpy_process(epsilon).view(-1,1)
    
    # Perform reduction
    # _log_reduction_for_sinkhorn(dp, k, edge, epsilon, debiasing=True)
    temp0 = _log_reduction_for_sinkhorn(
        dp=data_processor,
        k=0,
        edge=(0,1),
        epsilon=epsilon,
        debiasing=False,
    )

    temp1 = _log_reduction_for_sinkhorn(
        dp=data_processor,
        k=1,
        edge=(0,1),
        epsilon=epsilon,
        debiasing=False,
    )

    # NUMPY VERSION
    Fi = data_processor.data_dict[0]['f'].view(-1, 1).detach().cpu().numpy()
    # ---------- expected (NumPy) ----------
    if grid_type == "flat" or grid_type == "tensor":
        D = numpy_sqdist_matrix(Y.view(-1, 2).numpy(), X.view(-1, 2).numpy())
    elif grid_type == "tuple":
        D = numpy_sqdist_matrix(
           torch.cartesian_prod(*Y).numpy(),
           torch.cartesian_prod(*X).numpy(),
        )
    K = np.exp((Fi.reshape(-1,1) - 0.5 * D) / epsilon.item())
    expected = np.log(K.sum(0))

    print("expected bary:", expected.reshape(-1)[:10])
    print("temp1 bary:", temp1.detach().cpu().numpy().reshape(-1)[:10])
    print("temp1 bary:", temp0.detach().cpu().numpy().reshape(-1)[:10])

    assert np.allclose(expected.reshape(-1), temp0.detach().cpu().numpy().reshape(-1), rtol=1e-5, atol=1e-5)


    # epsilon must be positive scalar
    Fi = data_processor.data_dict[1]['f'].view(-1, 1).detach().cpu().numpy()
    # ---------- expected (NumPy) ----------
    K = np.exp((Fi.reshape(1,-1) - 0.5 * D) / epsilon.item())
    expected = np.log(K.sum(1))

    print('passes first')
    print("expected data:", expected.reshape(-1)[:10])
    print("mine data:", temp0.detach().cpu().numpy().reshape(-1)[:10])


    assert np.allclose(expected.reshape(-1), temp1.detach().cpu().numpy().reshape(-1), rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize(
    "n1, n2, members, m1, m2, L, grid_type",
    [
        (11, 10, 3, 5, 7, 0.9, "flat"),
        (8, 8, 3, 13, 8, 3.5, "tensor"),
        (12, 11, 3, 9, 9, 2.0, "tuple"),
    ],
)  # noqa: E501
def test_log_reduction_update_debiasingTRUE_against_numpy(
    n1, n2, members, m1, m2, L, grid_type
):
    np.random.seed(12313*members + n1 + n2 + m1 + m2)

    # Generate grid
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
    data_processor = generate_barycentredataprocessor(data, barycentre_grid=Y, potentials='f')

    # make up a debiasing potential 

    n_points = data_processor.data_dict[0]['f'].shape
    debiasing_potential = torch.rand(*n_points, 1, dtype=torch.double).squeeze()
    d = data_processor._torch_numpy_process(debiasing_potential)


    epsilon = 1 / max(np.sqrt(n1*n2), np.sqrt(m1*m2))
    epsilon = data_processor._torch_numpy_process(epsilon).view(-1,1)
    
    # Perform reduction
    # _log_reduction_for_sinkhorn(dp, k, edge, epsilon, debiasing=True)
    temp0 = _log_reduction_for_sinkhorn(
        dp=data_processor,
        k=0,
        d=d,
        edge=(0,1),
        epsilon=epsilon,
        debiasing=True,
    )

    temp1 = _log_reduction_for_sinkhorn(
        dp=data_processor,
        k=1,
        d=d,
        edge=(0,1),
        epsilon=epsilon,
        debiasing=True,
    )

    # NUMPY VERSION

    Fi = data_processor.data_dict[0]['f'].view(-1, 1).detach().cpu().numpy()
    # ---------- expected (NumPy) ----------
    if grid_type == "flat" or grid_type == "tensor":
        D = numpy_sqdist_matrix(Y.view(-1, 2).numpy(), X.view(-1, 2).numpy())
    elif grid_type == "tuple":
        D = numpy_sqdist_matrix(
           torch.cartesian_prod(*Y).numpy(),
           torch.cartesian_prod(*X).numpy(),
        )
    K = np.exp((Fi.reshape(-1,1) - 0.5 * D) / epsilon.item())
    expected = np.log(K.T @d.view(-1,1).detach().cpu().numpy())

    assert np.allclose(expected.reshape(-1), temp0.detach().cpu().numpy().reshape(-1), rtol=1e-5, atol=1e-5)

    # epsilon must be positive scalar
    Fi = data_processor.data_dict[1]['f'].view(-1, 1).detach().cpu().numpy()
    # ---------- expected (NumPy) ----------
    K = np.exp((Fi.reshape(1,-1) - 0.5 * D) / epsilon.item())
    expected = np.log(K.sum(1).reshape(-1,1) * d.view(-1,1).detach().cpu().numpy())

    assert np.allclose(expected.reshape(-1), temp1.detach().cpu().numpy().reshape(-1), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
