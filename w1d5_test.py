import w1d5_solution
from typing import cast, Callable

from sympy import symbols, Function, Add, Pow, Mul, Number
from utils import report

x, y, z, p = symbols("x y z p")
K = cast(Callable, Function("K"))

@report
def expand_exp_tests(student_expand_exp):
    assert student_expand_exp(K(x)).expand() == K(x)
    assert student_expand_exp(K(x**4)).expand() == K(x) ** 4 + 6 * K(x) ** 2 * K(x, x) + 4 * K(x) * K(
        x, x, x
    ) + 3 * K(x, x) ** 2 + K(x, x, x, x)
    assert student_expand_exp(K(x**2 * y**2)).expand() == K(x) ** 2 * K(y) ** 2 + K(x) ** 2 * K(y, y) + 4 * K(
        x
    ) * K(y) * K(x, y) + 2 * K(x) * K(x, y, y) + K(y) ** 2 * K(x, x) + 2 * K(y) * K(x, x, y) + K(x, x) * K(
        y, y
    ) + 2 * K(
        x, y
    ) ** 2 + K(
        x, x, y, y
    )
    assert student_expand_exp(K(x**3 * y)).expand() == K(x) ** 3 * K(y) + 3 * K(x) ** 2 * K(x, y) + 3 * K(x) * K(
        y
    ) * K(x, x) + 3 * K(x) * K(x, x, y) + K(y) * K(x, x, x) + 3 * K(x, x) * K(x, y) + K(x, x, x, y)
    assert student_expand_exp(K(x**2 * y * z)).expand() == K(x) ** 2 * K(y) * K(z) + K(x) ** 2 * K(y, z) + 2 * K(
        x
    ) * K(y) * K(x, z) + 2 * K(x) * K(z) * K(x, y) + 2 * K(x) * K(x, y, z) + K(y) * K(z) * K(x, x) + K(y) * K(
        x, x, z
    ) + K(
        z
    ) * K(
        x, x, y
    ) + K(
        x, x
    ) * K(
        y, z
    ) + 2 * K(
        x, y
    ) * K(
        x, z
    ) + K(
        x, x, y, z
    )
    assert student_expand_exp(K(x * p * y * z)).expand() == K(p) * K(x) * K(y) * K(z) + K(p) * K(x) * K(y, z) + K(
        p
    ) * K(y) * K(x, z) + K(p) * K(z) * K(x, y) + K(p) * K(x, y, z) + K(x) * K(y) * K(p, z) + K(x) * K(z) * K(p, y) + K(
        x
    ) * K(
        p, y, z
    ) + K(
        y
    ) * K(
        z
    ) * K(
        p, x
    ) + K(
        y
    ) * K(
        p, x, z
    ) + K(
        z
    ) * K(
        p, x, y
    ) + K(
        p, x
    ) * K(
        y, z
    ) + K(
        p, y
    ) * K(
        x, z
    ) + K(
        p, z
    ) * K(
        x, y
    ) + K(
        p, x, y, z
    )

@report
def bern_cumu_tests(bern_cumu):
    first_7 = [
        p,
        -p * (p - 1),
        p * (p - 1) * (2 * p - 1),
        -p * (p - 1) * (6 * p**2 - 6 * p + 1),
        p * (p - 1) * (2 * p - 1) * (12 * p**2 - 12 * p + 1),
        -p * (p - 1) * (120 * p**4 - 240 * p**3 + 150 * p**2 - 30 * p + 1),
        p * (p - 1) * (2 * p - 1) * (360 * p**4 - 720 * p**3 + 420 * p**2 - 60 * p + 1),
    ]

    # print(list(map(lambda x: x.expand(), bern_cumu(7))))
    # print(list(map(lambda x: x.expand(), first_7)))
    for correct, proposed in zip(first_7, bern_cumu(7)):
        # print(correct, proposed)
        assert correct.expand() == proposed.expand()

@report
def prod_rule_tests(student_expand_prods):
    print("Checking expand_exp_tests again...")
    expand_exp_tests(student_expand_prods)
    print("Further tests...")
    inputs = [K(x, y, z), K(x * y, x * y), K(x * y, x * z), K(x * y, x * y, z * z), K(x, y), K(x**2 * y, z)]
    for input in inputs:
        assert student_expand_prods(input).expand() == w1d5_solution.prod_rule(input).expand()

@report
def eval_gauss_bern_cumu_tests(student_egbc):
    pairs = [
        (K(x), Number(0)),
        (K(y), p),
        (K(x * x), Number(1)),
        (K(x * y, x), p),
        (K(x * y**2, x), p**2 + p * (1 - p)),
        (K(x * y * y, x * x), Number(0)),
        (K(x * y * y, x, y), 2 * p**2 * (1 - p) + p * (2 * p**2 - 3 * p + 1)),
    ]
    for (inp, out) in pairs:
        proposed = student_egbc(inp)
        assert (Number(proposed) if isinstance(proposed, int) else proposed.expand()) == out.expand()
