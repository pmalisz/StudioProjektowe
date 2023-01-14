class Singel:
    coefficients: list[float]
    probability: float

    def __init__(self, coefficients: list[float], probability: float):
        self.coefficients = coefficients.copy()
        self.a = self.coefficients[0]
        self.b = self.coefficients[1]
        self.c = self.coefficients[2]
        self.d = self.coefficients[3]
        self.e = self.coefficients[4]
        self.f = self.coefficients[5]
        self.probability = probability

    # TODO
    def mutate(self):
        raise NotImplementedError()
