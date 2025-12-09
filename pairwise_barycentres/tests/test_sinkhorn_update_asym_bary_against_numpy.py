import numpy as np
from scipy.spatial.distance import cdist
import torch
import pytest
from pwbarycentres import (
    asymmetric_sinkhorn_algorithm,
    generate_barycentredataprocessor,
    sinkhorn_update
)
import networkx as nx

torch.set_printoptions(precision=8)


# ==============================================================
# Testing sinkhorn udpate
# ==============================================================

@pytest.mark.parametrize(
    "n1, n2, members, L, grid_type",
    [
        (11, 11, 6, 1.0, "flat"),
        (11, 10, 7, 0.9, "flat"),
        (11, 12, 8, 0.9, "flat"),
        (8, 8, 4, 3.5, "tensor"),
        (9, 8, 3, 3.5, "tensor"),
        (8, 9, 2, 3.5, "tensor"),

    ],
)  # noqa: E501
def test_sinkhorn_update_with_same_grid_uniform_density_with_debiasing_again_torch(
    n1, n2, members, L, grid_type
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
    data = []

    for m in range(members):
        data.append([None, None])  # uniform density, grid will equal everywhere

    # generate the barycentre dataprocessor class which will store all objects
    data_processor = generate_barycentredataprocessor(data, barycentre_grid=X, grid=X)
    epsilon = np.sqrt(L*1/(n1*n2))
    epsilon = data_processor._torch_numpy_process(epsilon).view(-1,1)
    rho = 1.0
    d = data_processor._torch_numpy_process(torch.abs(torch.rand_like(data_processor.data_dict[0]['density']))+0.1)
    aprox = 'balanced'
    C = 0.5*torch.cdist(X.view(-1, 2), X.view(-1, 2), p=2).pow(2)

    # CHECK A FEW LOOPS:
    for _ in range(3):
        # bary update
        for edge in data_processor.graph.edges:

            # torch version
            a = data_processor.data_dict[edge[0]]['a'].view(-1, 1).cpu()
            b = data_processor.data_dict[edge[1]]['a'].view(-1, 1).cpu()
            density = data_processor.data_dict[edge[0]]['density'].view(-1, 1).cpu()

            torch_output = (torch.exp(
                -C / epsilon.cpu()
            )@torch.diag(d.view(-1).cpu())).T @ b

            # balance proxdiv
            torch_output = (1/(n1*n2)) / torch_output

            output = sinkhorn_update(
                data_processor, edge[0], edge, epsilon, rho, aprox, d, debiasing=True
            )

            print(torch_output.view(-1)[:10], output.view(-1)[:10].cpu()*n1*n2)

            assert torch.allclose(output.view(-1).cpu(), torch_output.view(-1), atol=1e-9)
            data_processor.data_dict[edge[0]]['a'] = output

        # potential update
        for edge in data_processor.graph.edges:

            # torch version
            a = data_processor.data_dict[edge[0]]['a'].view(-1, 1).cpu()
            b = data_processor.data_dict[edge[1]]['a'].view(-1, 1).cpu()
            density = data_processor.data_dict[edge[0]]['density'].view(-1, 1).cpu()

            torch_output = (torch.exp(
                -C / epsilon.cpu()
            )@torch.diag(d.view(-1).cpu())) @ a

            # balance proxdiv
            torch_output = (1/(n1*n2)) / torch_output

            output = sinkhorn_update(
                data_processor, edge[1], edge, epsilon, rho, aprox, d, debiasing=True
            )

            print(torch_output.view(-1)[:10], output.view(-1)[:10].cpu()*n1*n2)

            assert torch.allclose(output.view(-1).cpu(), torch_output.view(-1), atol=1e-9)
            data_processor.data_dict[edge[1]]['a'] = output


@pytest.mark.parametrize(
    "n1, n2, m1, m2, L, grid_type",
    [
        (20, 22, 21, 20, 1.0, "flat"),
        (50, 50, 50, 54, 1.0, "tensor"),
    ],
)  # noqa: E501
def test_sinkhorn_update_with_random_density_without_debiasing_again_torch(
    n1, n2, m1, m2, L, grid_type
):

    np.random.seed(n1 * n2 * m1 * m2)
    members = 1
    # tuple toggle for torch testing
    if grid_type == "flat":
        data = []
        Y = torch.cartesian_prod(
            torch.linspace(0, L, m1), torch.linspace(0, L, m2)
        ).type(torch.DoubleTensor)
        density = torch.abs(torch.rand(m1 * m2))
        # data.append([density / density.sum(), Y])  # central grid
        for m in range(members):  # member grids
            X = torch.cartesian_prod(
                torch.linspace(0, L, n1),
                torch.linspace(0, L, n2),
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
        # data.append([density / density.sum(), Y])  # central grid
        for m in range(members):
            X = torch.stack(
                torch.meshgrid(
                    torch.linspace(0, L, n1),
                    torch.linspace(0, L, n2),
                    indexing="ij",
                ),
                dim=-1,
            ).type(torch.DoubleTensor)
            density = torch.abs(torch.rand_like(X[..., 0]))
            data.append([density / density.sum(), X])

    # generate the barycentre dataprocessor class which will store all objects
    data_processor = generate_barycentredataprocessor(data=data, barycentre_grid=Y, grid=X)
    epsilon = max(np.sqrt(L*1/(n1*n2)), np.sqrt(L*1/(m1*m2)))
    epsilon = data_processor._torch_numpy_process(epsilon).view(-1,1)
    rho = 1.0
    aprox = 'balanced'
    C = 0.5*torch.cdist(X.view(-1, 2), Y.view(-1, 2), p=2).pow(2)

    # CHECK A FEW LOOPS:
    for _ in range(3):
        # bary update
        for edge in data_processor.graph.edges:

            # torch version
            a = data_processor.data_dict[edge[0]]['a'].view(-1, 1).cpu()
            b = data_processor.data_dict[edge[1]]['a'].view(-1, 1).cpu()
            density = data_processor.data_dict[edge[0]]['density'].view(-1, 1).cpu()
            print('SHAPES:')
            print(edge[0], a.shape, Y.shape)
            print(edge[1], b.shape, X.shape)

            torch_output = (torch.exp(
                -C / epsilon.cpu()
            )).T @ b

            # balance proxdiv
            torch_output = density / torch_output

            output = sinkhorn_update(
                data_processor, edge[0], edge, epsilon, rho, aprox, d=None, debiasing=False
            )

            print(torch_output.view(-1)[:10], output.view(-1)[:10].cpu()*n1*n2)

            assert torch.allclose(output.view(-1).cpu(), torch_output.view(-1), atol=1e-9)
            data_processor.data_dict[edge[0]]['a'] = output

        # potential update
        for edge in data_processor.graph.edges:

            # torch version
            a = data_processor.data_dict[edge[0]]['a'].view(-1, 1).cpu()
            b = data_processor.data_dict[edge[1]]['a'].view(-1, 1).cpu()
            density = data_processor.data_dict[edge[1]]['density'].view(-1, 1).cpu()

            torch_output = (torch.exp(
                -C / epsilon.cpu()
            )) @ a

            # balance proxdiv
            torch_output = density / torch_output

            output = sinkhorn_update(
                data_processor, edge[1], edge, epsilon, rho, aprox, d=None, debiasing=False
            )

            print(torch_output.view(-1)[:10], output.view(-1)[:10].cpu()*n1*n2)

            assert torch.allclose(output.view(-1).cpu(), torch_output.view(-1), atol=1e-9)
            data_processor.data_dict[edge[1]]['a'] = output


@pytest.mark.parametrize(
    "n1, n2, m1, m2, L, grid_type",
    [
        (20, 22, 21, 20, 1.0, "flat"),
        (50, 50, 50, 54, 1.0, "tensor"),
    ],
)  # noqa: E501
def test_sinkhorn_update_with_random_density_without_debiasing_again_torch_idkj(
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
        # data.append([density / density.sum(), Y])  # central grid
        for m in range(members):  # member grids
            X = torch.cartesian_prod(
                torch.linspace(0, L, n1),
                torch.linspace(0, L, n2),
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
        # data.append([density / density.sum(), Y])  # central grid
        for m in range(members):
            X = torch.stack(
                torch.meshgrid(
                    torch.linspace(0, L, n1),
                    torch.linspace(0, L, n2),
                    indexing="ij",
                ),
                dim=-1,
            ).type(torch.DoubleTensor)
            density = torch.abs(torch.rand_like(X[..., 0]))
            data.append([density / density.sum(), X])

    # generate the barycentre dataprocessor class which will store all objects
    data_processor = generate_barycentredataprocessor(data=data, barycentre_grid=Y, grid=X)
    epsilon = max(np.sqrt(L*1/(n1*n2)), np.sqrt(L*1/(m1*m2)))
    epsilon = data_processor._torch_numpy_process(epsilon).view(-1,1)
    rho = 1.0
    aprox = 'balanced'
    C = 0.5*torch.cdist(X.view(-1, 2), Y.view(-1, 2), p=2).pow(2)
    d = data_processor._torch_numpy_process(torch.abs(torch.rand_like(data_processor.data_dict[0]['density']))+0.1)
    K = torch.exp(-C / epsilon.cpu()) @ torch.diag(d.view(-1).cpu())  

    # CHECK A FEW LOOPS:
    for _ in range(3):
        # bary update
        for edge in data_processor.graph.edges:

            # torch version
            a = data_processor.data_dict[edge[0]]['a'].view(-1, 1).cpu()
            b = data_processor.data_dict[edge[1]]['a'].view(-1, 1).cpu()
            density = data_processor.data_dict[edge[0]]['density'].view(-1, 1).cpu()
            print('SHAPES:')
            print(edge[0], a.shape, Y.shape)
            print(edge[1], b.shape, X.shape)

            torch_output = K.T @ b

            # balance proxdiv
            torch_output = density / torch_output

            output = sinkhorn_update(
                data_processor, edge[0], edge, epsilon, rho, aprox, d, debiasing=True
            )

            print(torch_output.view(-1)[:10], output.view(-1)[:10].cpu()*n1*n2)

            assert torch.allclose(output.view(-1).cpu(), torch_output.view(-1), atol=1e-9)
            data_processor.data_dict[edge[0]]['a'] = output

        # potential update
        for edge in data_processor.graph.edges:

            # torch version
            a = data_processor.data_dict[edge[0]]['a'].view(-1, 1).cpu()
            b = data_processor.data_dict[edge[1]]['a'].view(-1, 1).cpu()
            density = data_processor.data_dict[edge[1]]['density'].view(-1, 1).cpu()

            torch_output = K @ a

            # balance proxdiv
            torch_output = density / torch_output

            output = sinkhorn_update(
                data_processor, edge[1], edge, epsilon, rho, aprox, d, debiasing=True
            )

            print(torch_output.view(-1)[:10], output.view(-1)[:10].cpu()*n1*n2)

            assert torch.allclose(output.view(-1).cpu(), torch_output.view(-1), atol=1e-9)
            data_processor.data_dict[edge[1]]['a'] = output

if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
