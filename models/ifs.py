import math
import numpy as np
from utils.img_processor import ImgProcessor

from models.singel import Singel


class Ifs:
    degree: int
    fitness: float
    singels: list[Singel]

    def __init__(self, singels: list[Singel]):
        self.singels = singels.copy()
        self.degree = len(singels)
        self.fitness = float("inf")
        self.normalize_singles_probabilities()

    def normalize_singles_probabilities(self):
        probabilities = [singel.probability for singel in self.singels]
        normalized_probabilities = [float(i) / sum(probabilities) for i in probabilities]

        for i in range(self.degree):
            self.singels[i].probability = round(normalized_probabilities[i], 2)

        diff = np.sum([singel.probability for singel in self.singels]) - 1
        if not math.isclose(diff, 0):
            while True:
                random_singel_index = np.random.randint(0, self.degree)
                if self.singels[random_singel_index].probability > diff:
                    self.singels[random_singel_index].probability -= diff
                    break

    def calculate_fitness(self, origin: np.ndarray, iterations: int, size: int):
        functions = np.ndarray(shape=(self.degree, 2, 3), dtype=float)
        for idx, singel in enumerate(self.singels):
            functions[idx] = [[singel.a, singel.b, singel.e], [singel.c, singel.d, singel.f]]

        probabilities = [singel.probability for singel in self.singels]

        gen = ImgProcessor.generate_fractal(iterations, size, functions, probabilities)
        # TODO

    # TODO
    def mutate(self):
        raise NotImplementedError()
