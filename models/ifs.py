import copy
import math

import numpy as np
from utils.img_processor import ImgProcessor

from models.singel import Singel


class Ifs:
    degree: int
    fitness: float
    singels: list[Singel]

    def __init__(self, singels: list[Singel]):
        self.singels = copy.deepcopy(singels)
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
                random_singel = self.singels[random_singel_index]
                if random_singel.probability > diff:
                    random_singel.probability -= diff
                    break

    # TODO img augmentation
    def calculate_fitness(self, origin: np.ndarray, iterations: int, size: int):
        functions = np.ndarray(shape=(self.degree, 2, 3), dtype=float)
        for idx, singel in enumerate(self.singels):
            functions[idx] = [[singel.a, singel.b, singel.e], [singel.c, singel.d, singel.f]]

        probabilities = [singel.probability for singel in self.singels]

        generated_img = ImgProcessor.generate_fractal(iterations, size, functions, probabilities)

        # TODO Na razie działa to z założeniem że rozmiar obrazu wejściowego jest taki jak obrazów generowanych
        self.fitness = size ** 2
        for x in range(size):
            for y in range(size):
                if generated_img[x][y] == origin[x][y]:
                    self.fitness -= 1

        print(self.fitness)

    # TODO
    def mutate(self):
        raise NotImplementedError()
