import numpy as np

from models.singel import Singel


class Ifs:
    degree: int
    fitness: float
    singles: list[Singel]

    def __init__(self, singels: list[Singel]):
        self.singles = singels.copy()
        self.degree = len(singels)
        self.fitness = float("inf")

    # TODO
    def calculate_fitness(self, origin: np.ndarray):
        raise NotImplementedError()

    # TODO
    def mutate(self):
        raise NotImplementedError()
