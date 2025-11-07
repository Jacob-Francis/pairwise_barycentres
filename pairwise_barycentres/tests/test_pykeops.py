import numpy as np
import pytest
import torch
from pwbarycentres.pykeops_formulas import chizat_marginals, chizat_reduction
import pykeops
# Import the functions you provided (adjust the import path as necessary)
# from your_module import chizat_reduction, chizat_marginals

# For demonstration, these imports are commented out. Uncomment and adjust above.

rtol = 1e-6
atol = 1e-8


# avoid scipy dependence
def numpy_sqdist_matrix(X, Y):
    """
    Return matrix D of squared distances between rows of X (n x d) and Y (m x d),
    shape (n, m).
    """
    # Using (x - y)^2 = ||x||^2 + ||y||^2 - 2 x.y
    X2 = np.sum(X**2, axis=1)[:, None]  # (n,1)
    Y2 = np.sum(Y**2, axis=1)[None, :]  # (1,m)
    XY = X @ Y.T                          # (n,m)
    D = X2 + Y2 - 2 * XY
    return D


@pytest.mark.parametrize("n_i,n_j,d", [(5, 7, 2), (1, 3, 2), (10, 10, 2)])
def test_chizat_reduction_matches_numpy(n_i, n_j, d):

    np.random.seed(12345 + n_i + n_j)
    Xi = np.random.uniform(size=(n_i, d)).astype(np.float64)
    Yj = np.random.uniform(size=(n_j, d)).astype(np.float64)

    # epsilon must be positive scalar
    epsilon = 1/np.sqrt(n_i*n_j)

    ai = np.random.rand(n_i).astype(np.float64)

    # ---------- expected (NumPy) ----------
    D = numpy_sqdist_matrix(Xi, Yj)  
    K = np.exp((-0.5 * D) / epsilon)  
    expected = K.T @ ai 

    # call with tensor - then detach and convert to numpy
   
    result = chizat_reduction(
        Xi=torch.tensor(Xi),
        Yj=torch.tensor(Yj),
        epsilon=torch.tensor(epsilon).view(-1,1),
        ai=torch.tensor(ai).view(-1,1)
    ).detach().cpu().view(-1).numpy()
    assert result.shape == expected.shape
    assert np.allclose(result, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("n_i,n_j,d", [(5, 7, 2), (1, 3, 2), (10, 10, 2)])
def test_chizat_marginal_matches_numpy(n_i, n_j, d):

    pykeops.clean_pykeops()
    np.random.seed(12313 + n_i + n_j)
    Xi = np.random.uniform(size=(n_i, d)).astype(np.float64)
    Yj = np.random.uniform(size=(n_j, d)).astype(np.float64)

    # epsilon must be positive scalar
    epsilon = 1/np.sqrt(n_i*n_j)

    ai = np.random.rand(n_i).astype(np.float64)
    bj = np.random.rand(n_j).astype(np.float64)

    # ---------- expected (NumPy) ----------
    D = numpy_sqdist_matrix(Xi, Yj)  
    K = np.exp((-0.5 * D) / epsilon)  
    expected = ((K.T @ ai )* bj)

    # call with tensor - then detach and convert to numpy
    result = chizat_marginals(
        Xi=torch.tensor(Xi),
        Yj=torch.tensor(Yj),
        epsilon=torch.tensor(epsilon).view(-1,1),
        ai=torch.tensor(ai).view(-1,1),
        bj=torch.tensor(bj).view(-1,1)
    ).detach().cpu().view(-1).numpy()

    assert result.shape == expected.shape
    assert np.allclose(result, expected, rtol=rtol, atol=atol)


def test_edge_cases_small_values():
    """
    Test a corner case: very small epsilon and near-zero distances to ensure numeric stability.
    """
    # two identical points -> distance 0
    Xi = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    Yj = Xi.copy()
    epsilon = 1e-6
    ai = np.array([1.0, 2.0], dtype=np.float64)
    bj = np.array([0.5, 1.5], dtype=np.float64)
    cost_const = 1.0

    # expected reduction: ai[i] * sum_j exp(-0.5 * ||xi - yj||^2 / eps)
    D = numpy_sqdist_matrix(Xi, Yj)
    K = np.exp((-0.5 * D) / epsilon)
    expected_reduction = ai[:, None] * K.sum(axis=1, keepdims=True)
    expected_marginals = (ai[:, None] * (np.exp((-0.5 * cost_const * D) / epsilon) * bj[None, :])).sum(axis=1, keepdims=True)

    out_r = chizat_reduction(
        Xi=torch.tensor(Xi),
        Yj=torch.tensor(Yj),
        epsilon=torch.tensor(epsilon).view(-1,1),
        ai=torch.tensor(ai).view(-1,1)
    ).detach().cpu().view(-1).numpy()
    out_m = chizat_marginals(
        Xi=torch.tensor(Xi),
        Yj=torch.tensor(Yj),
        epsilon=torch.tensor(epsilon).view(-1,1),
        ai=torch.tensor(ai).view(-1,1),
        bj=torch.tensor(bj).view(-1,1)
    ).detach().cpu().view(-1).numpy()

    np.testing.assert_allclose(np.asarray(out_r).ravel(), expected_reduction.ravel(), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(np.asarray(out_m).ravel(), expected_marginals.ravel(), rtol=1e-6, atol=1e-8)

if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
