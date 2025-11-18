from graph_dp import SinkhornDataProcessor
from .pykeops_formulaes import alpha_reduction_pykeops
from pwbarycentres import tensorise_f
import torch
import numpy as np

# ===================================================================================
# pointwise aprox things - should make these separate again
# ===================================================================================

def kl_prox(s, epsilon, rho, p):
    return s**(epsilon/(epsilon + rho)) * p**(rho/(epsilon + rho))

def kl_aprox(f, epsilon, rho):
    return rho*f/(epsilon + rho)

def balanced_aprox(f, epsilon, rho):
    return f

def tv_aprox(f, epsilon, rho):
    return torch.clip(f, min=-rho, max=rho)

def aprox_lse_update(f, epsilon, rho, aprox='balanced'):
    """
    f is the log-sum-exp value before aproximation
    """
    if aprox == 'kl':
        return kl_aprox(f, epsilon, rho)
    elif aprox == 'balanced':
        return balanced_aprox(f, epsilon, rho)
    elif aprox == 'tv':
        return tv_aprox(f, epsilon, rho)
    else:
        raise NotImplementedError("Only kl,tv and balanced aprox implemented")

def balanced_entropy(f, epsilon, rho):
    return f

def kl_entropy(f, epsilon, rho, tol=1e-13):
    """
    KL entropy term: rho * f * (log(f) - 1)
    with the convention that 0 * log(0) = 0.
    Entries <= tol are treated as 0 for stability.
    """
    out = torch.zeros_like(f)
    mask = f > tol
    if mask.any():
        fm = f[mask]
        out[mask] = rho * (fm * (torch.log(fm) - 1.0))
    return out

def balanced_prox(s, epsilon, rho, p):
    return p

def kl_proxdiv(s, epsilon, rho, p, u=None):
    
    if u is None:
        return (p/s)**(rho/(epsilon+rho))
    return (p/s)**(rho/(epsilon+rho)) * torch.exp(-u/(epsilon + rho))

def balanced_proxdiv(s, epsilon, rho, p, u=None):
    return p/s

def tv_prox(s, epsilon, rho, p):
    return torch.min(s*torch.exp(rho/epsilon), torch.max(s*torch.exp(-rho/epsilon), p))

def tv_proxdiv(s, epsilon, rho, p, u=None):
    if u is None:
        u = 0.0
    return torch.min(torch.exp((rho - u)/epsilon), torch.max(torch.exp((-rho + u)/epsilon), p/s))

def chizat_proxdiv_step(s, epsilon, rho, p, aprox='kl', u=None):
    """
    u is for kernel truncation purposes which may be useful later
    """
    if aprox == 'kl':
        return kl_proxdiv(s, epsilon, rho, p, u)
    elif aprox == 'balanced':
        return balanced_proxdiv(s, epsilon, rho, p, u)
    elif aprox == 'tv':
        return tv_proxdiv(s, epsilon, rho, p, u)
    else:
        raise NotImplementedError("Only kl and balanced aprox implemented")
    




def alpha_reduction(dp : SinkhornDataProcessor, j, k, epsilon, prod=True):
    """
    epsilon should already be on the correct device please

    Alpha reductions, alpha_(j,k) through edges; recusive reduction

    Equation (20) in Beier et al 2022

    """

    alpha = torch.ones_like(dp.data_dict[k]['f'])
    N = np.prod(alpha.shape)

    assert alpha.shape == dp.data_dict[(k, j)]["alpha"].shape

    for i in dp.graph.neighbors(k):
        if i == j:
            continue
        else:
            # Recursivly collect alpha variables along incoming edges to node k
            alpha *= alpha_reduction(dp, k, i, epsilon, prod=prod)


    # Now decide how to do the reduction based on whether we are using pykeops or tensorisation
    # We only store tensorisation grid on the ordered edges... though maybe i shoudl store it on
    if 'x1y1' in dp.data_dict[(j, k)] and 'x2y2' in dp.data_dict[(j, k)]:
        temp = _tensorised_alpha_reduction(
            dp.data_dict[(j, k)]['x1y1'],
            dp.data_dict[(j, k)]['x2y2'],
            alpha*dp.data_dict[k]['density'] if prod else alpha/N,
            dp.data_dict[(k)]['f'],
            epsilon,
        )
    elif 'grid' in dp.data_dict[k] and 'grid' in dp.data_dict[j]:
        temp = alpha_reduction_pykeops(
                Fi=dp.data_dict[k]['f'],
                Xi=dp.data_dict[k]['grid'],
                Yj=dp.data_dict[j]['grid'],
                E=epsilon,
                Mi=alpha.view(-1, 1)*dp.data_dict[k]['density'].view(-1, 1) if prod else alpha.view(-1, 1)/N
                )
    else:
        raise ValueError("No grid information found for alpha reduction")

    assert( temp.shape[0] == dp.data_dict[j]['f'].shape[0])
    
    # ToDo: update the dictionary - this is a recusive update which will overwrite previous values
    # which is probably added a lot of updates, but if I've calcauted a new value why wouldn't I 
    # keep it? I'm guessing I should - actually but for some this will be opnes!?
    return temp
    
def _tensorised_alpha_reduction(x1y1, x2y2, a, f, epsilon):
    return tensorise_f(
        torch.exp(-x1y1/epsilon),
        torch.exp(-x2y2/epsilon),
        a*torch.exp(f/epsilon)
    )

