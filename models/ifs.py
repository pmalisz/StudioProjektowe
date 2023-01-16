import copy

import numpy as np
from utils.math_helper import MathHelper
from utils.img_processor import ImgProcessor

from models.singel import Singel
import math

class Ifs:
    degree: int
    fitness: float
    singels: list[Singel]
    fractal: np.ndarray

    def __init__(self, singels: list[Singel]):
        self.singels = copy.deepcopy(singels)
        self.degree = len(singels)
        self.fitness = float("inf")
        self.normalize_singles_probabilities()

    def normalize_singles_probabilities(self):
        """
            IFS is created from universe of random singels with random probabilities, so those probabilities have to be
            normalized, so they sum up to 1 within given IFS
        """

        normalized_probabilities = MathHelper.normalize_probabilities([singel.probability for singel in self.singels])

        for i in range(self.degree):
            self.singels[i].probability = normalized_probabilities[i]

    # TODO img augmentation
    def calculate_fitness(self, origin: np.ndarray, iterations: int, size: int):
        functions = np.ndarray(shape=(self.degree, 2, 3), dtype=float)
        for idx, singel in enumerate(self.singels):
            functions[idx] = [[singel.a, singel.b, singel.e], [singel.c, singel.d, singel.f]]

        probabilities = [singel.probability for singel in self.singels]

        self.fractal = ImgProcessor.generate_fractal(iterations, size, functions, probabilities)

        # TODO Na razie działa to z założeniem że rozmiar obrazu wejściowego jest taki jak obrazów generowanych
        self.fitness = size**2
        for x in range(size):
            for y in range(size):
                if self.fractal[x][y] == origin[x][y]:
                    self.fitness -= 1

    # TODO
    def mutate(self):
        raise NotImplementedError()
