import torch
import numpy as np
from torchnumpyprocess import TorchNumpyProcessing


"""
I need this class to handle data and given grids, and store them efficientcy. If all the data shares the same grid,
then it should only store it once. 
If they each live on a different grid then I need to store them separately,
Though if tensorsaion is possible I also need to io store all the different small memeory kerenals - which 
may be many different grids...

I'll focus on the non-graph case for now - though maybe desiging both at the same time ie better.

ToDo: Generalise so that not all the data has the same type - i.e. is tuple and all points?

"""
class BarycentreDataProcessor(TorchNumpyProcessing):
    def __init__(self, data_dict, graph, grid=None, set_fail=False, cuda_device=None, pykeops=True, free_grids=True):
        """Provide a python dictionary with keys being the index of the data and keys, density, grid.
        If given a grid assume they all share one grid. It will ignore any keys inside the dictionary
        for the same grid

        Based off of the given grid(s) decide if it is possible to tensorise. 

        We assume PyKeOps is always avaliable

        Graph (this could be a graph definind a pair wise version no? though can you have separate edges)
        Parameters
        ----------
        data_dict : dict
            dictionary with keys being the index of the data and keys, density, grid.
        graph : networkx graph
            graph defining the connections between the data points.
        grid : (N, 2), (n1, n2, 2), ((n1), (n2)), optional
            If provided then we assume all data lives on the same grid, by default None
        set_fail : bool, optional
            see parent class, by default False
        cuda_device : int, optional
            see parent class, by default None
        pykeops : bool, optional
            If True then we assume PyKeOps is available and use it for tensorisation, by default True
        """

        # old import
        super().__init__(set_fail, cuda_device)
        self.graph = graph
        self.data_dict = data_dict
        self.pykeops = pykeops

        # Run processing of the graph edges
        self.build_edges(grid=grid)
        self._density_processing()
        if free_grids:
            self.free_grid_memory()
    
    def _process_grids(self, edge, grid1, grid2):
        """

        Parameters
        ----------
        edge: tuple
            tuple i,j -- i.e. (3,4) defining the edge being considered
            from grid1 to grid 2.
        grid1 : (N, 2), (n1, n2, 2), ((n1), (n2))
            _description_
        grid2 : (M, 2), (m1, m2, 2), ((m1), (m2))
            _description_
        """

        # Toggle
        self.data_dict[edge] = {}

        if isinstance(grid1, tuple) and isinstance(grid2, tuple):
            self.data_dict[edge]['x1y1'] = (
                0.5
                * torch.cdist(
                    self._clone_process(grid1[0], non_blocking=True).view(
                        -1, 1
                    ),
                    self._clone_process(grid2[0], non_blocking=True).view(
                        -1, 1
                    ),
                )
                ** 2
            )
            self.data_dict[edge]['x2y2'] = (
                0.5
                * torch.cdist(
                    self._clone_process(grid1[1], non_blocking=True).view(
                        -1, 1
                    ),
                    self._clone_process(grid2[1], non_blocking=True).view(
                        -1, 1
                    ),
                )
                ** 2
            )
        elif len(grid1.shape)==3 and len(grid2.shape)==3:
            print('Any meshgrid inputs are assumed to be creating with indexing="ij":' \
            'For equally sized meshes this does not matter but for different sizes it does.')
            n1, n2, n3 = grid1.shape
            m1, m2, m3 = grid2.shape

            assert n3==2 and m3==2, "We assume 2D points"

            # Calculate cost matrices - the indexing works
            # because torch cdist eliminats the common axis which will have the same values.
            self.data_dict[edge]['x1y1'] = (
                0.5
                * torch.cdist(
                    self._clone_process(grid1[:n1, 0], non_blocking=True),
                    self._clone_process(grid2[:m1, 0], non_blocking=True),
                )
                ** 2
            )
            self.data_dict[edge]['x2y2'] = (
                0.5
                * torch.cdist(
                    self._clone_process(grid1[0, :n2], non_blocking=True),
                    self._clone_process(grid2[0, :m2], non_blocking=True),
                )
                ** 2
            )
            # Prioritise tensoration
        elif self.pykeops == True:
            # Need to process for PyKeOps - otherwise can delete?
            self.data_dict[edge[0]]['grid'] = self._clone_process(grid1, non_blocking=True)
            self.data_dict[edge[1]]['grid'] = self._clone_process(grid2, non_blocking=True)
        else:
            # I'm not sure I'll ever use this
            raise NotImplementedError("Creating the full dense kernel is not supported")
    
    def _edge_density_processing(self, edge, density1, density2):
        try:
            n1,n2 = self.data_dict[edge[0]]['grid'].shape
            m1,m2 = self.data_dict[edge[1]]['grid'].shape
            
            assert n2 ==2 and m2==2, "We assume 2D points"
            
            # for the potentials we drop the two
            n2 = m2 = 1
        except (KeyError, AttributeError, ValueError):
            n1, m1 = self.data_dict[edge]['x1y1'].shape
            n2, m2 = self.data_dict[edge]['x2y2'].shape
        
        # overwrite with correct verison
        self.data_dict[edge[0]]['density'] = self._process_inputs(density1, n1, n2)
        self.data_dict[edge[1]]['density'] = self._process_inputs(density2, m1, m2)
    
    def _process_inputs(self, points, n, m, const=1):
        """
        Processes densities or points or potentials, with default 'None' values as ones*constant. Or convert input to torch type
        """
        if points is None:
            weights = const * torch.ones((n, m)).type(self.dtype) / (n * m)
        else:
            weights = self._clone_process(points, non_blocking=True)
            weights = weights.view(n, m)

        return weights

    def _density_processing(self):
        # go around all edges of the graph and check if i can tensorise

        for edge in self.graph.edges:
            density_i = self.data_dict[edge[0]]['density']
            density_j = self.data_dict[edge[1]]['density']
            self._edge_density_processing(edge, density_i, density_j)

    def build_edges(self, grid=None):
        # go around all edges of the graph and check if i can tensorise

        if grid is None: # generate per edge
            for edge in self.graph.edges:
                
                grid_i = self.data_dict[edge[0]]['grid']
                grid_j = self.data_dict[edge[1]]['grid']
                self._process_grids(edge, grid_i, grid_j)
        else: # They're sharing the grid
            edge_list = list(self.graph.edges)
            edge0 = edge_list.pop()
            self._process_grids(edge0, grid, grid)

            # Point to the same grid for all data
            if isinstance(grid, tuple) or len(grid.shape)==3:
                x1y1 = self.data_dict[edge0]['x1y1']
                x2y2 = self.data_dict[edge0]['x2y2']

                for edge in edge_list:
                    self.data_dict[edge]={}
                    self.data_dict[edge]['x1y1'] = x1y1 
                    self.data_dict[edge]['x2y2'] = x2y2

                # check pointing to the same object for efficient memory use
                for e in list(self.graph.edges):
                    assert self.data_dict[e]['x1y1'] is x1y1
                    assert self.data_dict[e]['x2y2'] is x2y2
            elif self.pykeops == True:
                # Point to one grid
                shared_grid = self.data_dict[edge0[0]]['grid']
                for edge in list(self.graph.edges):
                    self.data_dict[edge] = {}
                    self.data_dict[edge[0]]['grid'] = shared_grid
                    self.data_dict[edge[1]]['grid'] = shared_grid

                # check pointing to the same object for efficient memory use
                for e in list(self.graph.edges):
                    assert self.data_dict[e[0]]['grid'] is shared_grid
                    assert self.data_dict[e[1]]['grid'] is shared_grid

    def free_grid_memory(self):
        # bool switch to delete otherwise we need to keep the grid -
        # We have actually assumed that each piece of data assumes the same grid struture.
        for k in self.graph.nodes():
            del_graph = True
            # print(self.data_dict.keys())
            for n in self.graph.neighbors(k):
                # We use an undirected graph and so only need to check one way
                if 'x1y1' in self.data_dict[(k,n) if k<n else (n,k)]:
                    continue
                else:
                    # There is an edge without tensorsiation - so we need to keep the grid
                    del_graph = False
            
            if del_graph:
                del self.data_dict[k]['grid']
       