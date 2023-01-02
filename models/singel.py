class Singel:
    coefficients: list[float]

    def __init__(self, coefficients: list[float]):
        self.coefficients = coefficients.copy()

    # TODO
    def mutate(self):
        raise NotImplementedError()
