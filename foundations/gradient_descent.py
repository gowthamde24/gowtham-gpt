def minimize_quadratic(iterations: int, learning_rate: float, init: int) -> float:
    """Minimize f(x) = x^2 via gradient descent, returning the final x."""
    minimizer = init

    for _ in range(iterations):
        derivate = 2 * minimizer
        minimizer = minimizer - learning_rate * derivate

    return round(minimizer, 5)
