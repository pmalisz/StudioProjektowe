import random


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

    def mutate(self):
        random_coefficient_idx = random.randint(0, 5)
        if random_coefficient_idx < 4:
            self.coefficients[random_coefficient_idx] = random.uniform(-0.99, 1)
        else:
            self.coefficients[random_coefficient_idx] = random.uniform(-2, 2) # TODO change fixed values

