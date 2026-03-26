import numpy as np
from scipy.spatial.distance import cdist
import torch
import pytest
from pwbarycentres import (
    asymmetric_sinkhorn_algorithm,
    generate_barycentredataprocessor,
    sinkhorn_update,
    balanced_barycentre_updates,
    debiasing_dual_potential_update

)
import networkx as nx

torch.set_printoptions(precision=8)

@pytest.mark.parametrize(
    "n1, n2, m1, m2, L, grid_type",
    [
        (20, 22, 21, 20, 1.0, "flat"),
        (50, 50, 50, 54, 1.0, "tensor"),
    ],
)  # noqa: E501
def test_d_sinkhorn_update_with_random_density_with_debiasing_fixed_agast_torch_withexplicit_d(
    n1, n2, m1, m2, L, grid_type
):

    np.random.seed(n1 * n2 * m1 * m2)
    members = 3
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
    d = torch.abs(torch.rand_like(data_processor.data_dict[0]['a']))
    rho = 1.0
    aprox = 'balanced'
    C = 0.5*torch.cdist(X.view(-1, 2), Y.view(-1, 2), p=2).pow(2)
    K = torch.exp(-C / epsilon.cpu()) @ torch.diag(d.view(-1).cpu())

    # CHECK A FEW LOOPS:
    for _ in range(10):

        # potential update
        for edge in data_processor.graph.edges:

            # torch version
            a = data_processor.data_dict[edge[0]]['a'].view(-1, 1).cpu()
            b = data_processor.data_dict[edge[1]]['a'].view(-1, 1).cpu()
            density = data_processor.data_dict[edge[1]]['density'].view(-1).cpu()

            torch_output = K @ a

            # balance proxdiv
            torch_output = density / torch_output.view(-1)

            output = sinkhorn_update(
                data_processor, edge[1], edge, epsilon, rho, aprox, d, debiasing=True
            )

            assert torch.allclose(output.view(-1).cpu(), torch_output.view(-1), atol=1e-9)
            data_processor.data_dict[edge[1]]['a'] = output

        # Update the barycentre
        true_barycentre = torch.ones_like(data_processor.data_dict[edge[0]]['density']).cpu().view(-1)
        for e1, e2, weight in data_processor.graph.edges.data('weight'):
            b = data_processor.data_dict[e2]['a'].view(-1, 1).cpu()
            s = K.T @ b
            true_barycentre *= (s.cpu()**(weight.cpu())).view(-1)

        # implemented update
        barycentre = balanced_barycentre_updates(data_processor, d, epsilon)

        assert torch.allclose(barycentre.view(-1).cpu(), true_barycentre.view(-1), atol=1e-5), 'Error = ' + str(torch.norm(barycentre.view(-1).cpu() - true_barycentre.view(-1).cpu(), p=2).item())
        
        for edge in data_processor.graph.edges:
            data_processor.data_dict[edge[0]]['density'] = barycentre
        
        # bary update
        for edge in data_processor.graph.edges:

            # torch version
            a = data_processor.data_dict[edge[0]]['a'].view(-1, 1).cpu()
            b = data_processor.data_dict[edge[1]]['a'].view(-1, 1).cpu()
            # density = data_processor.data_dict[edge[0]]['density'].view(-1, 1).cpu()

            torch_output = K.T @ b

            # balance proxdiv
            torch_output = true_barycentre.view(-1) / torch_output.view(-1)

            output = sinkhorn_update(
                data_processor, edge[0], edge, epsilon, rho, aprox, d, debiasing=True
            )

            assert torch.allclose(output.view(-1).cpu(), torch_output.view(-1), atol=1e-9)
            data_processor.data_dict[edge[0]]['a'] = output

@pytest.mark.parametrize(
    "n1, n2, m1, m2, L, grid_type",
    [
        (20, 22, 21, 20, 1.0, "flat"),
        (50, 50, 50, 54, 1.0, "tensor"),
    ],
)  # noqa: E501
def test_d_sinkhorn_update_with_random_density_with_debiasing_fixed_agast_torch_withoutexplicit_d(
    n1, n2, m1, m2, L, grid_type
):

    np.random.seed(n1 * n2 * m1 * m2)
    members = 3
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
    d = torch.abs(torch.rand_like(data_processor.data_dict[0]['a']))
    rho = 1.0
    aprox = 'balanced'
    C = 0.5*torch.cdist(X.view(-1, 2), Y.view(-1, 2), p=2).pow(2)
    K = torch.exp(-C / epsilon.cpu())  #@ torch.diag(d.view(-1).cpu())

    # CHECK A FEW LOOPS:
    for _ in range(10):

        # potential update
        for edge in data_processor.graph.edges:

            # torch version
            a = data_processor.data_dict[edge[0]]['a'].view(-1, 1).cpu()
            b = data_processor.data_dict[edge[1]]['a'].view(-1, 1).cpu()
            density = data_processor.data_dict[edge[1]]['density'].view(-1).cpu()

            torch_output = K @ a

            # balance proxdiv
            torch_output = density / torch_output.view(-1)

            output = sinkhorn_update(
                data_processor, edge[1], edge, epsilon, rho, aprox, d=None, debiasing=False
            )

            assert torch.allclose(output.view(-1).cpu(), torch_output.view(-1), atol=1e-9)
            data_processor.data_dict[edge[1]]['a'] = output

        # Update the barycentre
        true_barycentre = d.view(-1).cpu()
        for e1, e2, weight in data_processor.graph.edges.data('weight'):
            b = data_processor.data_dict[e2]['a'].view(-1, 1).cpu()
            s = K.T @ b
            true_barycentre *= (s.cpu()**(weight.cpu())).view(-1)

        # implmeneted udpate
        barycentre = balanced_barycentre_updates(data_processor, d, epsilon)
        print('SUMS', barycentre.sum().item(), true_barycentre.sum().item())
        assert torch.allclose(barycentre.view(-1).cpu(), true_barycentre.view(-1), atol=1e-9)
        
        for edge in data_processor.graph.edges:
            data_processor.data_dict[edge[0]]['density'] = barycentre
        
        # bary update
        for edge in data_processor.graph.edges:

            # torch version
            a = data_processor.data_dict[edge[0]]['a'].view(-1, 1).cpu()
            b = data_processor.data_dict[edge[1]]['a'].view(-1, 1).cpu()
            # density = data_processor.data_dict[edge[0]]['density'].view(-1, 1).cpu()

            torch_output = K.T @ b

            # balance proxdiv
            torch_output = true_barycentre.view(-1) / torch_output.view(-1)

            output = sinkhorn_update(
                data_processor, edge[0], edge, epsilon, rho, aprox, d=None, debiasing=False
            )

            assert torch.allclose(output.view(-1).cpu(), torch_output.view(-1), atol=1e-9)
            data_processor.data_dict[edge[0]]['a'] = output
        
    
@pytest.mark.parametrize(
    "n1, n2, m1, m2, L, grid_type",
    [
        (20, 22, 21, 20, 1.0, "flat"),
        (50, 50, 50, 54, 1.0, "tensor"),
    ],
)  # noqa: E501
def test_d_sinkhorn_update_with_random_density_without_debiasing_again_torch(
    n1, n2, m1, m2, L, grid_type
):

    np.random.seed(n1 * n2 * m1 * m2)
    members = 3
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
    for _ in range(10):

        # potential update
        for edge in data_processor.graph.edges:

            # torch version
            a = data_processor.data_dict[edge[0]]['a'].view(-1, 1).cpu()
            b = data_processor.data_dict[edge[1]]['a'].view(-1, 1).cpu()
            density = data_processor.data_dict[edge[1]]['density'].view(-1).cpu()

            torch_output = (torch.exp(
                -C / epsilon.cpu()
            )) @ a

            # balance proxdiv
            torch_output = density / torch_output.view(-1)

            output = sinkhorn_update(
                data_processor, edge[1], edge, epsilon, rho, aprox, d=None, debiasing=False
            )

            assert torch.allclose(output.view(-1).cpu(), torch_output.view(-1), atol=1e-9)
            data_processor.data_dict[edge[1]]['a'] = output

        # Update the barycentre
        true_barycentre = torch.ones_like(data_processor.data_dict[edge[0]]['density']).cpu().view(-1)
        K = torch.exp(-C / epsilon.cpu())
        for e1, e2, weight in data_processor.graph.edges.data('weight'):
            b = data_processor.data_dict[e2]['a'].view(-1, 1).cpu()
            s = K.T @ b
            true_barycentre *= (s.cpu()**(weight.cpu())).view(-1)

        # implmeneted udpate
        barycentre = balanced_barycentre_updates(data_processor, torch.ones_like(data_processor.data_dict[edge[0]]['density']), epsilon)
        assert torch.allclose(barycentre.view(-1).cpu(), true_barycentre.view(-1), atol=1e-9)
        
        for edge in data_processor.graph.edges:
            data_processor.data_dict[edge[0]]['density'] = barycentre
        
        # bary update
        for edge in data_processor.graph.edges:

            # torch version
            a = data_processor.data_dict[edge[0]]['a'].view(-1, 1).cpu()
            b = data_processor.data_dict[edge[1]]['a'].view(-1, 1).cpu()
            # density = data_processor.data_dict[edge[0]]['density'].view(-1, 1).cpu()

            torch_output = (torch.exp(
                -C / epsilon.cpu()
            )).T @ b

            # balance proxdiv
            torch_output = true_barycentre.view(-1) / torch_output.view(-1)

            output = sinkhorn_update(
                data_processor, edge[0], edge, epsilon, rho, aprox, d=None, debiasing=False
            )

            assert torch.allclose(output.view(-1).cpu(), torch_output.view(-1), atol=1e-9)
            data_processor.data_dict[edge[0]]['a'] = output

@pytest.mark.parametrize(
    "n1, n2, m1, m2, L, grid_type",
    [
        (20, 22, 21, 20, 1.0, "flat"),
        # (50, 50, 50, 54, 1.0, "tensor"), not doing ebcause its neds to be processed more
    ],
)  # noqa: E501
def test_d_update_convergence_test(
    n1, n2, m1, m2, L, grid_type
):

    np.random.seed(n1 * n2 * m1 * m2)
    members = 3
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

    
    # generate the barycentre dataprocessor class which will store all objects
    data_processor = generate_barycentredataprocessor(data=data, barycentre_grid=Y, grid=X)
    epsilon = max(np.sqrt(L*1/(n1*n2)), np.sqrt(L*1/(m1*m2)))
    epsilon = data_processor._torch_numpy_process(epsilon).view(-1,1)
    d = torch.abs(torch.rand_like(data_processor.data_dict[0]['a']))
    dtemp = d.clone().cpu().view(-1, 1)
    rho = 1.0
    aprox = 'balanced'
    C = 0.5*torch.cdist(Y.view(-1, 2), Y.view(-1, 2), p=2).pow(2)
    K = torch.exp(-C / epsilon.cpu()).cpu()
    barycentre = torch.abs(torch.rand_like(data_processor.data_dict[0]['density']))

    for _ in range(50):
        d = debiasing_dual_potential_update(data_processor, d, barycentre, epsilon)
        dtemp = torch.sqrt(dtemp.cpu() * barycentre.view(-1, 1).cpu() / (K @ dtemp.cpu()))
        assert torch.allclose(dtemp.view(-1).cpu(), d.view(-1).cpu(), atol=1e-9)

    # check constriant
    err = torch.norm(d.cpu().view(-1) - barycentre.cpu().view(-1)  / (K@d.cpu()).view(-1) , p=2).item()
    assert err < 1e-9

    err = torch.norm( dtemp.view(-1) - barycentre.view(-1).cpu() / (K @ dtemp.cpu()).view(-1), p=2).item()
    assert err < 1e-9

if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
