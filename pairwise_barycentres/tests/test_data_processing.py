import numpy as np
from scipy.spatial.distance import cdist
import torch
import pytest
from pwbarycentres import BarycentreDataProcessor
import networkx as nx

torch.set_printoptions(precision=8)

@pytest.mark.parametrize(
    "n1, n2, L, grid_type",
    [
        (2, 2, 0.9, 'flat'),
        (5, 6, 3.5, 'tensor'),
        (12, 11,6.0, 'tuple'),
    ],
)  # noqa: E501
def test_class_with_same_grid_uniform_density(n1, n2, L, grid_type):
    
    if grid_type == 'flat':
        X = torch.cartesian_prod(torch.linspace(0, L, n1), torch.linspace(0, L, n2)).type(torch.DoubleTensor)
    elif grid_type == 'tensor':
        X = torch.stack(torch.meshgrid(
            torch.linspace(0, L, n1),
            torch.linspace(0, L, n2),
            indexing='ij'
        ), dim=-1).type(torch.DoubleTensor)
    elif grid_type == 'tuple':
        X = (torch.linspace(0, L, n1), torch.linspace(0, L, n2))
    data_dict = {}
    edges = [(1,2), (2,3), (3,4)]
    G = nx.Graph()
    G.add_edges_from(edges)

    for k in G.nodes():
        data_dict[k] = {}
        data_dict[k]['grid'] = None
        data_dict[k]['density'] = None

    # Grid is flat and we're using pykeops so should just add a grid variable
    bcp = BarycentreDataProcessor(
        graph=G,
        data_dict=data_dict,
        grid=X,
        cuda_device='cpu',
    )

    assert bcp.data_dict == data_dict

    # Check that the shared grid object is shared and equal to the right thing
    if grid_type == 'flat':
        shared_grid = bcp.data_dict[1]['grid']
        for nodes in G.nodes():
            assert bcp.data_dict[nodes]['grid'] is shared_grid
        
        assert (shared_grid==X).all()

    # check densities are unform
    if isinstance(X, tuple):
        expected_density = torch.ones((X[0].shape[0], X[1].shape[0]), dtype=torch.double)
        expected_density /= expected_density.sum()
    else:
        expected_density = torch.ones((X.shape[:-1]), dtype=torch.double)
        expected_density /= expected_density.sum()

    for nodes in G.nodes():
        density = bcp.data_dict[nodes]['density']
        assert np.isclose(density.sum().item(), 1.0)
        assert torch.isclose(density, expected_density, atol=1e-12, rtol=1e-12).all()

@pytest.mark.parametrize(
    "int_list, L, graph_type",
    [
        ([2, 2, 3, 3, 4, 3, 2, 2], 0.9, 'line'),
        ([5, 6, 7, 8, 9, 10], 3.5, 'line'),
        ([12, 11, 10], 6.0, 'star'),
        ([6, 6, 6, 6, 6], 10.0, 'star'),
    ],
)  # noqa: E501
def test_class_with_meshgrid_uniform_density(int_list, L, graph_type):
    
    data_dict = {}
    if graph_type == 'line':
        edges = [(k, k+1) for k in range(len(int_list)-1)]
    elif graph_type == 'star':
        edges = [(1, k) for k in range(2, len(int_list)+1)]
    
    G = nx.Graph()
    G.add_edges_from(edges)

    # create meshgrids
    for k in G.nodes():
        data_dict[k] = {}
        data_dict[k]['grid'] = torch.stack(torch.meshgrid(
            torch.linspace(0, L, int_list[(k)%len(int_list)]),
            torch.linspace(0, L, int_list[(k+1)%len(int_list)]),
            indexing='ij'
        ), dim=-1).type(torch.DoubleTensor)

        data_dict[k]['density'] = None

    # Grid is flat and we're using pykeops so should just add a grid variable
    bcp = BarycentreDataProcessor(
        graph=G,
        data_dict=data_dict,
        cuda_device='cpu',
        free_grids=False
    )

    assert bcp.data_dict == data_dict

    for edges in G.edges():
        grid_i = data_dict[edges[0]]['grid']
        n1, n2, _ = grid_i.shape
        grid_j = data_dict[edges[1]]['grid']
        m1, m2, _ = grid_j.shape

        # check grids processed correctly
        x1y1 = bcp.data_dict[edges]['x1y1']
        x2y2 = bcp.data_dict[edges]['x2y2']

        dists_x = torch.cdist(
            grid_i[:n1, 0],
            grid_j[:m1, 0]
        )**2 / 2.0
        dists_y = torch.cdist(
            grid_i[0, :n2],
            grid_j[0, :m2]
        )**2 / 2.0

        assert torch.isclose(x1y1, dists_x, atol=1e-12, rtol=1e-12).all()
        assert torch.isclose(x2y2, dists_y, atol=1e-12, rtol=1e-12).all()


@pytest.mark.parametrize(
    "int_list, L, graph_type",
    [
        ([2, 2, 3, 3, 4, 3, 2, 2], 0.9, 'line'),
        ([5, 6, 7, 8, 9, 10], 3.5, 'line'),
        ([12, 11, 10], 6.0, 'star'),
        ([6, 6, 6, 6, 6], 10.0, 'star'),
    ],
)  # noqa: E501
def test_class_with_tuple_uniform_density(int_list, L, graph_type):
    
    data_dict = {}
    if graph_type == 'line':
        edges = [(k, k+1) for k in range(len(int_list)-1)]
    elif graph_type == 'star':
        edges = [(1, k) for k in range(2, len(int_list)+1)]
    
    G = nx.Graph()
    G.add_edges_from(edges)

    # create meshgrids
    for k in G.nodes():
        data_dict[k] = {}
        data_dict[k]['grid'] = (
            torch.linspace(0, L, int_list[(k)%len(int_list)]),
            torch.linspace(0, L, int_list[(k+1)%len(int_list)]),
        )

        data_dict[k]['density'] = None

    # Grid is flat and we're using pykeops so should just add a grid variable
    bcp = BarycentreDataProcessor(
        graph=G,
        data_dict=data_dict,
        cuda_device='cpu',
        free_grids=False
    )

    for edges in G.edges():
        grid_i = data_dict[edges[0]]['grid']
        grid_j = data_dict[edges[1]]['grid']

        # check grids processed correctly
        x1y1 = bcp.data_dict[edges]['x1y1']
        x2y2 = bcp.data_dict[edges]['x2y2']

        dists_x = torch.cdist(
            grid_i[0].view(-1,1).to(dtype=torch.float64),
            grid_j[0].view(-1,1).to(dtype=torch.float64)
        )**2 / 2.0
        dists_y = torch.cdist(
            grid_i[1].view(-1,1).to(dtype=torch.float64),
            grid_j[1].view(-1,1).to(dtype=torch.float64)
        )**2 / 2.0

        assert torch.isclose(x1y1, dists_x, atol=1e-12, rtol=1e-12).all()
        assert torch.isclose(x2y2, dists_y, atol=1e-12, rtol=1e-12).all()


@pytest.mark.parametrize(
    "int_list, L, graph_type",
    [
        ([2, 2, 3, 3, 4, 3, 2, 2], 0.9, 'line'),
        ([5, 6, 7, 8, 9, 10], 3.5, 'line'),
        ([12, 11, 10], 6.0, 'star'),
        ([6, 6, 6, 6, 6], 10.0, 'star'),
    ],
)  # noqa: E501
def test_class_with_flat_uniform_density(int_list, L, graph_type):
    
    data_dict = {}
    if graph_type == 'line':
        edges = [(k, k+1) for k in range(len(int_list)-1)]
    elif graph_type == 'star':
        edges = [(1, k) for k in range(2, len(int_list)+1)]
    
    G = nx.Graph()
    G.add_edges_from(edges)

    # create meshgrids
    for k in G.nodes():
        data_dict[k] = {}
        data_dict[k]['grid'] = torch.cartesian_prod(
            torch.linspace(0, L, int_list[(k)%len(int_list)]),
            torch.linspace(0, L, int_list[(k+1)%len(int_list)]),
        )

        data_dict[k]['density'] = None

    # Grid is flat and we're using pykeops so should just add a grid variable
    bcp = BarycentreDataProcessor(
        graph=G,
        data_dict=data_dict,
        cuda_device='cpu'
    )

    for edges in G.edges():

        # check is still does have grid attribute
        assert 'grid' in bcp.data_dict[edges[0]]
        assert 'grid' in bcp.data_dict[edges[1]]

        # check grids processed correctly
        assert not 'x1y1' in bcp.data_dict[edges]
        assert not 'x2y2' in bcp.data_dict[edges]
         




if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
