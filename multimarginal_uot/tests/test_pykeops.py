import pytest
import torch
from mmuot import alpha_reduction_pykeops


@pytest.mark.parametrize(
    "num_i, num_j",
    [
        (10, 15),
        (20, 30),
        (13, 11),
        (5,5),
    ],
)
def test_alpha_reduction_pykeops(num_i, num_j):
    # Generate random input data
    F = torch.randn(num_i, 1, dtype=torch.float64)
    X = torch.randn(num_i, 2, dtype=torch.float64)
    Y = torch.randn(num_j, 2, dtype=torch.float64)
    E = torch.tensor([0.5], dtype=torch.float64).view(-1,1)
    m = abs(torch.randn(num_i, 1, dtype=torch.float64))

    # Run your function
    temp = alpha_reduction_pykeops(F, X, Y, E, m)

    # Reference computation
    ref = (
        torch.exp((F.view(-1, 1) - torch.cdist(X, Y) ** 2 / 2) / E) * m.view(-1, 1)
    ).sum(0)

    print(temp/ref)
    print('shapes', temp.shape, ref.shape)
    # Assert that results are close
    assert torch.allclose(
        temp, ref.view(-1, 1), atol=1e-8
    ), f"Failed at size: num_i={num_i}, num_j={num_j}"


# @pytest.mark.parametrize(
#     "num_i, num_j, dimension",
#     [
#         (10, 15, 2),
#         (20, 30, 2),
#         (50, 80, 2),
#     ],
# )
# def test_kl_pilogpi_term_reduction_pykeops(num_i, num_j, dimension):
#     # Generate random input data
#     F = torch.randn(num_i, 1, dtype=torch.float64)
#     G = torch.randn(num_j, 1, dtype=torch.float64)
#     X = torch.randn(num_i, dimension, dtype=torch.float64)
#     Y = torch.randn(num_j, dimension, dtype=torch.float64)
#     E = torch.tensor([0.5], dtype=torch.float64)
#     T = torch.tensor([0.5], dtype=torch.float64)
#     m = abs(torch.randn(num_i, 1, dtype=torch.float64))

#     truth = kl_pilogpi_term_reduction_pykeops(F, G, X, Y, E, T, m).squeeze()

#     # Reference computation
#     ref = (
#         ((F.view(-1, 1) + G.view(1, -1) - torch.cdist(X, Y) ** 2 * T / 2) / E)
#         * torch.exp(
#             (F.view(-1, 1) + G.view(1, -1) - torch.cdist(X, Y) ** 2 * T / 2) / E
#         )
#         * m.view(-1, 1)
#     ).sum(0)

#     # Assert that results are close
#     assert torch.allclose(
#         truth, ref, atol=1e-8
#     ), f"Failed at difference {torch.linalg.norm(ref-truth)}"

# @pytest.mark.parametrize(
#     "num_i, num_j, dimension",
#     [
#         (10, 15, 2),
#         (20, 30, 2),
#         (50, 80, 2),
#     ],
# )
# def test_cost_pi_term_reduction_pykeops(num_i, num_j, dimension):
#     # Generate random input data
#     F = torch.randn(num_i, 1, dtype=torch.float64)
#     G = torch.randn(num_j, 1, dtype=torch.float64)
#     X = torch.randn(num_i, dimension, dtype=torch.float64)
#     Y = torch.randn(num_j, dimension, dtype=torch.float64)
#     E = torch.tensor([0.5], dtype=torch.float64)
#     T = torch.tensor([0.5], dtype=torch.float64)
#     m = abs(torch.randn(num_i, 1, dtype=torch.float64))

#     truth = cost_pi_term_reduction_pykeops(F, G, X, Y, E, T, m).squeeze()

#     # Reference computation
#     ref = (
#         (torch.cdist(X, Y) ** 2 * T / 2)
#         * torch.exp(
#             (F.view(-1, 1) + G.view(1, -1) - torch.cdist(X, Y) ** 2 * T / 2) / E
#         )
#         * m.view(-1, 1)
#     ).sum(0)

#     # Assert that results are close
#     assert torch.allclose(
#         truth, ref, atol=1e-8
#     ), f"Failed at difference {torch.linalg.norm(ref-truth)}"

# @pytest.mark.parametrize(
#     "num_i, num_j",
#     [
#         (10, 15),
#         (20, 30),
#         (50, 80),
#     ],
# )
# def test_kl_pilogcost_term(num_i, num_j):
#     # Generate random input data
#     A = torch.randn(num_i, 1, dtype=torch.float64)
#     B = torch.randn(num_j, 1, dtype=torch.float64)
#     X = torch.randn(num_i, 2, dtype=torch.float64)
#     Y = torch.randn(num_j, 2, dtype=torch.float64)
#     E = torch.tensor([0.5], dtype=torch.float64)
#     T = torch.tensor([0.5], dtype=torch.float64)

#     truth = kl_pilogcost_term(A, B, X, Y, E, T).squeeze()

#     # Reference computation
#     ref = (A.view(-1, 1) * B.view(1, -1) * torch.exp(
#             (torch.cdist(X, Y) ** 2 * T / 2) / E
#         )* (torch.cdist(X, Y) ** 2 * T / 2) /E
#     ).sum(0)

#     # Assert that results are close
#     assert torch.allclose(
#         truth, ref, atol=1e-8
#     ), f"Failed at difference {torch.linalg.norm(ref-truth)}"



if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main([__file__]))
