class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        pass

        minimizer = init

        for _ in range(iterations):
            derivate = 2 * minimizer
            minimizer = minimizer - learning_rate * derivate

        return round(minimizer,5)
