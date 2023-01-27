import random
import numpy as np


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
            self.coefficients[random_coefficient_idx] = random.uniform(-2, 2)  # TODO change fixed values

    @staticmethod
    def cross_singels(singel1: 'Singel', singel2: 'Singel') -> 'Singel':
        point_of_division = random.randint(0, 5)
        new_coefficients = singel1.coefficients[:point_of_division] + singel2.coefficients[point_of_division:]
        new_probability = (singel1.probability + singel2.probability) / 2
        return Singel(new_coefficients, new_probability)

    def __repr__(self):
        return f"a: {np.round(self.a, 2)} " \
               f"b: {np.round(self.b, 2)} " \
               f"c: {np.round(self.c, 2)} " \
               f"d: {np.round(self.d, 2)} " \
               f"e: {np.round(self.e, 2)} " \
               f"f: {np.round(self.f, 2)} " \
               f"| probability: {np.round(self.probability, 2)}"
