from .utils import alpha_reduction_pykeops, alpha_reduction
from .mmuot_sinkhorn_graph_reductions import (
    sinkhorn_update,
    mmuot_sinkhorn_loop,
    generate_mmuotdataprocessor_star_graph,
    generate_mmuot_debiasing_dp
)
from .mmuot_costing import mmuot_dual_cost, mmuot_marginal_j, mmuot_marginals
