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
    def __init__(
        self,
        data_dict,
        graph,
        grid=None,
        density=None,
        set_fail=False,
        cuda_device=None,
        pykeops=True,
        free_grids=True,
        verbose=False,
    ):
        """Provide a python dictionary with keys being the index of the data and keys, density, grid.
        If given a grid assume they all share one grid. It will ignore any keys inside the dictionary
        for the same grid

        GRAPH SHOULD HAVE THE WEIGHTS ACCOCIATED TO EACH EDGE

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
        cell_areas:
            Leb elements a list per entry/ or without it creates its own
            Actually it creates its own then overwrites this if given some
        """

        # old import
        super().__init__(set_fail, cuda_device)
        self.graph = graph
        self.data_dict = data_dict
        self.pykeops = pykeops
        self.verbose = verbose

        assert len(self.graph.nodes) == len(
            self.data_dict
        ), "Data dict and graph nodes do not match in size"

        # Run processing of the graph edges
        self.build_edges(grid=grid)
        self._density_processing(density=density)
        self.process_cell_areas()

        # Useful attributes
        self.num_of_edges = len(self.graph.edges)

        self.process_graph_weights()

        if free_grids:
            self.free_grid_memory()

    def process_graph_weights(self):
        # Ensure weight on correct device
        for edge in self.graph.edges:
            if "weight" not in self.graph[edge[0]][edge[1]]:
                self.graph[edge[0]][edge[1]]["weight"] = self._torch_numpy_process(
                    torch.tensor(1.0/self.num_of_edges).type(
                    self.dtype
                )
                ).view(-1, 1)
            else:
                self.graph[edge[0]][edge[1]]["weight"] = self._torch_numpy_process(
                    self.graph[edge[0]][edge[1]]["weight"]
                ).view(-1, 1)

    def _cell_area_tuple(self, grid):
        """
        grid in shape ((n1), (n2))
        assume regular
        """
        cell_area = (grid[0][1] - grid[0][0]) * (grid[1][1] - grid[1][0])

        return self._torch_numpy_process(cell_area)
    
    def _cell_area_tensor(self, grid1):
        """
        grid in shape (n1, n2, 2)
        """
        x = grid1[..., 0]  # (n1, n2)
        y = grid1[..., 1]  # (n1, n2)
        dx_val = (x[1, 0] - x[0, 0])
        dy_val = (y[0, 1] - y[0, 0])
        cell_area = dx_val * dy_val
        return self._torch_numpy_process(cell_area)
    
    def _cell_area_flat(self, grid1):
        """
        grid in shape (n1*n2, 2)
        """
        x = torch.unique(grid1[..., 0], sorted=True)  # (n1*n2)
        y = torch.unique(grid1[..., 1], sorted=True)  # (n1*n2)
        dx_val = (x[1] - x[0])
        dy_val = (y[1] - y[0])
        cell_area = dx_val * dy_val
        return self._torch_numpy_process(cell_area)

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
            self.data_dict[edge]["x1y1"], self.data_dict[edge]["x2y2"] = (
                self._cost_for_tuple(grid1, grid2)
            )

            # cell areas: if not given use approximation for regular grid
            if self.data_dict[edge[0]].get("cell_areas", None) is None:
                self.data_dict[edge[0]]["cell_areas"] = self._cell_area_tuple(grid1)
            if self.data_dict[edge[1]].get("cell_areas", None) is None:
                self.data_dict[edge[1]]["cell_areas"] = self._cell_area_tuple(grid2)

        elif len(grid1.shape) == 3 and len(grid2.shape) == 3:
            if self.verbose:
                print(
                    'Any meshgrid inputs are assumed to be creating with indexing="ij":'
                    "For equally sized meshes this does not matter but for different sizes it does."
                )
            n1, n2, n3 = grid1.shape
            m1, m2, m3 = grid2.shape

            assert n3 == 2 and m3 == 2, "We assume 2D points"

            # Calculate cost matrices - the indexing works
            # because torch cdist eliminats the common axis which will have the same values.
            self.data_dict[edge]["x1y1"], self.data_dict[edge]["x2y2"] = (
                self._cost_for_meshgrid(grid1, grid2, n1, n2, m1, m2)
            )

            # cell areas - if not given use approximation for regular grid
            if self.data_dict[edge[0]].get("cell_areas", None) is None:
                self.data_dict[edge[0]]["cell_areas"] = self._cell_area_tensor(grid1)
            if self.data_dict[edge[1]].get("cell_areas", None) is None:
                self.data_dict[edge[1]]["cell_areas"] = self._cell_area_tensor(grid2)

            # Prioritise tensoration
        elif self.pykeops == True and not (
            isinstance(grid1, tuple) and isinstance(grid2, tuple)
        ):

            assert grid1.shape[1] == 2 and grid2.shape[1] == 2, "We assume 2D points"

            # Need to process for PyKeOps - otherwise can delete?
            self.data_dict[edge[0]]["grid"] = self._clone_process(
                grid1, non_blocking=True
            )
            self.data_dict[edge[1]]["grid"] = self._clone_process(
                grid2, non_blocking=True
            )

            # cell areas - if not given use approximation for regular grid
            if self.data_dict[edge[0]].get("cell_areas", None) is None:
                self.data_dict[edge[0]]["cell_areas"] = self._cell_area_flat(grid1)
            if self.data_dict[edge[1]].get("cell_areas", None) is None:
                self.data_dict[edge[1]]["cell_areas"] = self._cell_area_flat(grid2)

        else:
            # I'm not sure I'll ever use this
            raise NotImplementedError("Creating the full dense kernel is not supported")

    def _edge_density_processing(self, edge, density1, density2):
        try:
            n1, n2 = self.data_dict[edge[0]]["grid"].shape
            m1, m2 = self.data_dict[edge[1]]["grid"].shape

            assert n2 == 2 and m2 == 2, "We assume 2D points"

            # for the potentials we drop the two
            n2 = m2 = 1
        except (KeyError, AttributeError, ValueError):
            n1, m1 = self.data_dict[edge]["x1y1"].shape
            n2, m2 = self.data_dict[edge]["x2y2"].shape

        # overwrite with correct verison
        self.data_dict[edge[0]]["density"] = self._process_inputs(density1, n1, n2, cell_areas=self.data_dict[edge[0]].get("cell_areas", None))
        self.data_dict[edge[1]]["density"] = self._process_inputs(density2, m1, m2, cell_areas=self.data_dict[edge[1]].get("cell_areas", None))

    def _process_inputs(self, points, n, m, cell_areas=None):
        """
        Processes densities or points or potentials, with default 'None' values as ones*constant. Or convert input to torch type
        """
        if points is None:
            if cell_areas is not None:
                weights = self._torch_numpy_process(
                        torch.ones((n, m)).type(self.dtype)
                    )/ (n*m) / cell_areas 
            else: weights = self._torch_numpy_process(torch.ones((n, m)).type(self.dtype))
            # divide so that the total mass is 1 
        else:
            weights = self._clone_process(points, non_blocking=True)
            weights = weights.view(n, m)

        return weights

    def _density_processing(self, density=None):
        # go around all edges of the graph and check if i can tensorise

        if density is None:
            for edge in self.graph.edges:
                density_i = self.data_dict[edge[0]]["density"]
                density_j = self.data_dict[edge[1]]["density"]
                self._edge_density_processing(edge, density_i, density_j)
        else:
            # assume there is only one density shared on every node
            edge_list = list(self.graph.edges)
            edge0 = edge_list.pop()
            self._edge_density_processing(edge0, density, density)

            shared_density = self.data_dict[edge0[0]]["density"]
            for node in self.graph.nodes:
                self.data_dict[node]["density"] = shared_density

            for edges in list(self.graph.edges):
                assert self.data_dict[edges[0]]["density"] is shared_density
                assert self.data_dict[edges[1]]["density"] is shared_density

    def build_edges(self, grid=None):
        # go around all edges of the graph and check if i can tensorise

        if grid is None:  # generate per edge
            for edge in self.graph.edges:

                grid_i = self.data_dict[edge[0]]["grid"]
                grid_j = self.data_dict[edge[1]]["grid"]
                self._process_grids(edge, grid_i, grid_j)
        else:  # They're sharing the grid
            edge_list = list(self.graph.edges)
            edge0 = edge_list.pop()
            self._process_grids(edge0, grid, grid)

            # share cell_areas
            for edge in list(self.graph.edges):
                self.data_dict[edge[0]]["cell_areas"] = self.data_dict[edge0[0]]["cell_areas"]
                self.data_dict[edge[1]]["cell_areas"] = self.data_dict[edge0[1]]["cell_areas"]

            # Point to the same grid for all data
            if isinstance(grid, tuple) or len(grid.shape) == 3:
                x1y1 = self.data_dict[edge0]["x1y1"]
                x2y2 = self.data_dict[edge0]["x2y2"]

                for edge in edge_list:
                    self.data_dict[edge] = {}
                    self.data_dict[edge]["x1y1"] = x1y1
                    self.data_dict[edge]["x2y2"] = x2y2

                # check pointing to the same object for efficient memory use
                for e in list(self.graph.edges):
                    assert self.data_dict[e]["x1y1"] is x1y1
                    assert self.data_dict[e]["x2y2"] is x2y2
            elif self.pykeops == True:
                # Point to one grid
                shared_grid = self.data_dict[edge0[0]]["grid"]
                for edge in list(self.graph.edges):
                    self.data_dict[edge] = {}
                    self.data_dict[edge[0]]["grid"] = shared_grid
                    self.data_dict[edge[1]]["grid"] = shared_grid

                # check pointing to the same object for efficient memory use
                for e in list(self.graph.edges):
                    assert self.data_dict[e[0]]["grid"] is shared_grid
                    assert self.data_dict[e[1]]["grid"] is shared_grid


    def free_grid_memory(self):
        # bool switch to delete otherwise we need to keep the grid -
        # We have actually assumed that each piece of data assumes the same grid struture.
        for k in self.graph.nodes():
            del_graph = True
            # print(self.data_dict.keys())
            for n in self.graph.neighbors(k):
                # We use an undirected graph and so only need to check one way
                if "x1y1" in self.data_dict[(k, n) if k < n else (n, k)]:
                    continue
                else:
                    # There is an edge without tensorsiation - so we need to keep the grid
                    del_graph = False

            if del_graph:
                del self.data_dict[k]["grid"]

    def _cost_for_meshgrid(self, grid1, grid2, n1, n2, m1, m2):
        return (
            0.5
            * torch.cdist(
                self._clone_process(grid1[:n1, 0], non_blocking=True),
                self._clone_process(grid2[:m1, 0], non_blocking=True),
            )
            ** 2,
            0.5
            * torch.cdist(
                self._clone_process(grid1[0, :n2], non_blocking=True),
                self._clone_process(grid2[0, :m2], non_blocking=True),
            )
            ** 2,
        )

    def _cost_for_tuple(self, grid1, grid2):
        return (
            0.5
            * torch.cdist(
                self._clone_process(grid1[0], non_blocking=True).view(-1, 1),
                self._clone_process(grid2[0], non_blocking=True).view(-1, 1),
            )
            ** 2,
            0.5
            * torch.cdist(
                self._clone_process(grid1[1], non_blocking=True).view(-1, 1),
                self._clone_process(grid2[1], non_blocking=True).view(-1, 1),
            )
            ** 2,
        )

    def process_cell_areas(self, ):
        '''
        data_dict should have key per node 'cell_areas' which is either a float or a vector or none, or needs to be assined.
        if its a vector then store at the right node and check its size matches the grid size
        also convert to torch. 
        Check if its all unique if it is a tensor.

        If none given assine 1/N where N is the number of points in the grid.
        '''
        # just checking that the density and shapes make
        for node in self.graph.nodes:
            cell_areas = self.data_dict[node].get("cell_areas", None)
    
            if cell_areas is None:
                raise ValueError(f"Cell areas must be provided for node {node} in the data dict")
            elif isinstance(cell_areas, (float, int)):
                self.data_dict[node]['cell_areas'] = self._torch_numpy_process(cell_areas)
            elif isinstance(cell_areas, torch.Tensor) and cell_areas.ndim == 0:
                self.data_dict[node]['cell_areas'] = self._torch_numpy_process(cell_areas)
            else:
                # check it has the right shape for the density
                assert cell_areas.shape == self.data_dict[node]['density'].shape, f"Cell areas shape {cell_areas.shape} does not match density shape {self.data_dict[node]['density'].shape} for node {node}"
                self.data_dict[node]['cell_areas'] = self._clone_process(cell_areas)

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#  Sinkhorn Data Processor
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------


class SinkhornDataProcessor(BarycentreDataProcessor):
    def __init__(
        self,
        data_dict,
        graph,
        potentials="f",
        grid=None,
        density=None,
        set_fail=False,
        cuda_device=None,
        pykeops=True,
        free_grids=True,
    ):
        super().__init__(
            data_dict, graph, grid, density, set_fail, cuda_device, pykeops, free_grids
        )

        # Which potentials do you need? a or f where a = exp(f/eps)
        if potentials == "a":
            self._initial_dual_potentials_ab()
        elif potentials == "f":
            self._initial_dual_potentials_fg()
        else:
            raise ValueError("potentials must be either 'a' or 'f'")

    def _initial_dual_potentials_ab(self):
        # attach correct potential per node
        for node in self.graph.nodes:
            self.data_dict[node]["a"] = self._torch_numpy_process(
                torch.ones_like(self.data_dict[node]["density"])
            )

    def _initial_dual_potentials_fg(self):

        # attach correct potential per node
        for node in self.graph.nodes:
            self.data_dict[node]["f"] = self._torch_numpy_process(
                torch.zeros_like(self.data_dict[node]["density"])
            )
