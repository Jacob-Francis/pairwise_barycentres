import torch
import numpy as np
from tensorisation import Tensorisation


"""
I need this class to handle data and given grids, and store them efficientcy. If all the data shares the same grid,
then it should only store it once. 
If they each live on a different grid then I need to store them separately,
Though if tensorsaion is possible I also need to io store all the different small memeory kerenals - which 
may be many different grids...

I'll focus on the non-graph case for now - though maybe desingin both at the same time ie better.
 
"""
class BarycentreDataProcessor(Tensorisation):
    def __init__(self, data_dict, graph, grid=None, set_fail=False, cuda_device=None, pykeops=True):
        """Provide a python dictionary with keys being the index of the data and keys, density, grid.
        If given a grid assume they all share one grid. It will ignore any keys inside the dictionary
        for the same grid

        Based off of the given grid(s) decide if it is possible to tensorise. 

        We assume PyKeOps is always avaliable

        Graph (this could be a graph definind a pair wise version no? though can you have separate edges)
        Parameters
        ----------
        data : _type_
            _description_
        """

        # old import
        super().__init__(set_fail, cuda_device)
        self.graph = graph
        self.data_dict = data_dict
        self.pykeops = pykeops

        # Run processing of the graph edges
        self.build_edges(grid=grid)
    
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
        using_pykeops = False

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
            using_pykeops = True
        else:
            # I'm not sure I'll ever use this
            raise NotImplementedError("Creating the full dense kernel is not supported")

        if not using_pykeops:
            # Free up memory
            del self.data_dict[edge[0]]['grid']
            del self.data_dict[edge[1]]['grid']
    
    def density_processing(self, edge, density1, density2):
        try:
            n1,n2 = self.data_dict[edge[0]]['grid'].shape
            m1,m2 = self.data_dict[edge[1]]['grid'].shape
            
            assert n2 ==2 and m2==2, "We assume 2D points"
            
            # for the potentials we drop the two
            n2 = m2 = 1
        except KeyError:
            n1, m1 = self.data_dict[edge[0]]['x1y1'].shape
            n2, m2 = self.data_dict[edge[0]]['x2y2'].shape
        
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

    def build_edges(self, grid=None):
        # go around all edges of the graph and check if i can tensorise

        if grid is None:
            for edge in self.graph.edges:
                grid_i = self.data_dict[edge[0]]['grid']
                grid_j = self.data_dict[edge[1]]['grid']
                self._process_grids(edge, grid_i, grid_j)
        else:
            edge_list = list(self.graph.edges)
            edge0 = edge_list.pop()
            self._process_grids(edge, grid, grid)

            x1y1 = self.data_dict[edge0]['x1y1']
            x2y2 = self.data_dict[edge0]['x2y2']

            # Point to the same grid for all data
            if len(grid.shape)==3 or isinstance(grid, tuple):
                for edge in edge_list:
                    self.data_dict[edge]['x1y1'] = x1y1 
                    self.data_dict[edge]['x2y2'] = x2y2

            # check pointing to the same object for efficient memory use
            for e in edge_list:
                assert self.data_dict[e]['x1y1'] is x1y1
                assert self.data_dict[e]['x2y2'] is x2y2

          