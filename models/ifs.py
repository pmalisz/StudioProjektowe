import numpy as np
import utils.img_processor

from models.singel import Singel


class Ifs:
    degree: int
    fitness: float
    singles: list[Singel]

    def __init__(self, singels: list[Singel]):
        self.singles = singels.copy()
        self.degree = len(singels)
        self.fitness = float("inf")

    def calculate_fitness(self, origin: np.ndarray, iterations: int, size: int, ):
        gen = utils.generate_fractal(iterations, size, )
        raise NotImplementedError()

    # TODO
    def mutate(self):
        raise NotImplementedError()
