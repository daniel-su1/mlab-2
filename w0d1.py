import math
from einops import rearrange, repeat, reduce
import torch as t

def assert_all_equal(actual: t.Tensor, expected: t.Tensor) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert (actual == expected).all(), f"Value mismatch, got: {actual}"
    print("Passed!")


def assert_all_close(actual: t.Tensor, expected: t.Tensor, rtol=1e-05, atol=0.0001) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert t.allclose(actual, expected, rtol=rtol, atol=atol)
    print("Passed!")


def rearrange_1():
    """Return the following tensor using only torch.arange and einops.rearrange:

    [[3, 4],
     [5, 6],
     [7, 8]]
    """

    return rearrange(t.arange(3, 9), "(h w) -> h w", h=3, w=2)


expected = t.tensor([[3, 4], [5, 6], [7, 8]])
assert_all_equal(rearrange_1(), expected)


def rearrange_2():
    """Return the following tensor using only torch.arange and einops.rearrange:

    [[1, 2, 3],
     [4, 5, 6]]
    """

    return rearrange(t.arange(1, 7), "(h w) -> h w", h=2, w=3)


assert_all_equal(rearrange_2(), t.tensor([[1, 2, 3], [4, 5, 6]]))


def rearrange_3():
    """Return the following tensor using only torch.arange and einops.rearrange:

    [[[1], [2], [3], [4], [5], [6]]]
    """
    return rearrange(t.arange(1, 7), "(h w) -> 1 h w", h=6, w=1)


assert_all_equal(rearrange_3(), t.tensor([[[1], [2], [3], [4], [5], [6]]]))


def temperatures_average(temps: t.Tensor):
    """Return the average temperature for each week.

    temps: a 1D temperature containing temperatures for each day.
    Length will be a multiple of 7 and the first 7 days are for the first week, second 7 days for the second week, etc.

    You can do this with a single call to reduce.
    """
    assert len(temps) % 7 == 0
    return reduce(temps, "(h 7) -> h", "mean")


temps = t.tensor([71, 72, 70, 75, 71, 72, 70, 68, 65, 60, 68, 60, 55, 59, 75, 80, 85, 80, 78, 72, 83], dtype=t.float32)
expected = t.tensor([71.5714, 62.1429, 79.0])
assert_all_close(temperatures_average(temps), expected)


def temperatures_differences(temps: t.Tensor):
    """For each day, subtract the average for the week the day belongs to.

    temps: as above
    """
    assert len(temps) % 7 == 0
    avg = repeat(temperatures_average(temps), "w -> (w 7)")
    return temps - avg


expected = t.tensor(
    [
        -0.5714,
        0.4286,
        -1.5714,
        3.4286,
        -0.5714,
        0.4286,
        -1.5714,
        5.8571,
        2.8571,
        -2.1429,
        5.8571,
        -2.1429,
        -7.1429,
        -3.1429,
        -4.0,
        1.0,
        6.0,
        1.0,
        -1.0,
        -7.0,
        4.0,
    ]
)
actual = temperatures_differences(temps)
assert_all_close(actual, expected)


def temperatures_normalized(temps: t.Tensor):
    """For each day, subtract the weekly average and divide by the weekly standard deviation.

    temps: as above

    Pass torch.std to reduce.
    """

    difs = temperatures_differences(temps)
    stds = repeat(reduce(temps, "(h 7) -> h", t.std), "w -> (w 7)")

    return t.div(difs, stds)


expected = t.tensor(
    [
        -0.3326,
        0.2494,
        -0.9146,
        1.9954,
        -0.3326,
        0.2494,
        -0.9146,
        1.1839,
        0.5775,
        -0.4331,
        1.1839,
        -0.4331,
        -1.4438,
        -0.6353,
        -0.8944,
        0.2236,
        1.3416,
        0.2236,
        -0.2236,
        -1.5652,
        0.8944,
    ]
)
actual = temperatures_normalized(temps)
assert_all_close(actual, expected)

def batched_dot_product_nd(a: t.Tensor, b: t.Tensor):
    """Return the batched dot product of a and b, where the first dimension is the batch dimension.

    That is, out[i] = dot(a[i], b[i]) for i in 0..len(a).
    a and b can have any number of dimensions greater than 1.

    a: shape (b, i_1, i_2, ..., i_n)
    b: shape (b, i_1, i_2, ..., i_n)

    Returns: shape (b, )

    Use torch.einsum. You can use the ellipsis "..." in the einsum formula to represent an arbitrary number of dimensions.
    """
    assert a.shape == b.shape
    return t.einsum("b...,b... -> b", a, b)


actual = batched_dot_product_nd(t.tensor([[1, 1, 0], [0, 0, 1]]), t.tensor([[1, 1, 0], [1, 1, 0]]))
expected = t.tensor([2, 0])
assert_all_equal(actual, expected)

actual2 = batched_dot_product_nd(t.arange(12).reshape((3, 2, 2)), t.arange(12).reshape((3, 2, 2)))
expected2 = t.tensor([14, 126, 366])
assert_all_equal(actual2, expected2)

def identity_matrix(n: int):
    """Return the identity matrix of size nxn.

    Don't use torch.eye or similar.

    Hint: you can do it with arange, rearrange, and ==.
    Bonus: find a different way to do it.
    """
    assert n >= 0
    return (rearrange(t.arange(n), "i->i 1") == t.arange(n)).float()


assert_all_equal(identity_matrix(3), t.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
assert_all_equal(identity_matrix(0), t.zeros((0, 0)))

def sample_distribution(probs: t.Tensor, n: int):
    """Return n random samples from probs, where probs is a normalized probability distribution.

    probs: shape (k,) where probs[i] is the probability of event i occurring.
    n: number of random samples

    Return: shape (n,) where out[i] is an integer indicating which event was sampled.

    Use torch.rand and torch.cumsum to do this without any explicit loops.

    Note: if you think your solution is correct but the test is failing, try increasing the value of n.
    """
    assert abs(probs.sum() - 1.0) < 0.001
    assert (probs >= 0).all()
    return (t.rand(n, 1) > t.cumsum(probs, dim=0)).sum(dim=-1)


n = 10000000
probs = t.tensor([0.05, 0.1, 0.1, 0.2, 0.15, 0.4])
freqs = t.bincount(sample_distribution(probs, n)) / n
assert_all_close(freqs, probs, rtol=0.001, atol=0.001)

print(t.rand(n, 1))
print(t.cumsum(probs, dim=0))
print((t.rand(n, 1) > t.cumsum(probs, dim=0)).sum(dim=-1))