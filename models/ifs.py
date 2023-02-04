import copy
import math
import random
import sys
import cv2

import numpy as np
from utils.math_helper import MathHelper
from utils.img_processor import ImgProcessor

from models.singel import Singel


class Ifs:
    degree: int
    fitness: int
    singels: list[Singel]
    fractal: np.ndarray

    def __init__(self, singels: list[Singel]):
        self.singels = copy.deepcopy(singels)
        self.degree = len(singels)
        self.fitness = sys.maxsize
        self.normalize_singles_probabilities()

    def normalize_singles_probabilities(self):
        """
            IFS is created from universe of random singels with random probabilities, so those probabilities have to be
            normalized, so they sum up to 1 within given IFS
        """

        normalized_probabilities = MathHelper.normalize_probabilities([singel.probability for singel in self.singels])

        for i in range(self.degree):
            self.singels[i].probability = normalized_probabilities[i]

    def calculate_fitness(self, origin: np.ndarray, iterations: int, size: int):
        functions = np.ndarray(shape=(self.degree, 2, 3), dtype=float)
        for idx, singel in enumerate(self.singels):
            functions[idx] = [[singel.a, singel.b, singel.e], [singel.c, singel.d, singel.f]]

        probabilities = [singel.probability for singel in self.singels]

        if not math.isclose(np.sum(probabilities), 1):
            self.fitness = size ** 2
            return

        self.fractal = ImgProcessor.generate_fractal(iterations, size, functions, probabilities)

        # self.calculate_base_fitness(size, origin)
        self.calculate_better_fitness(size, origin)

    def calculate_base_fitness(self, size, origin):
        self.fitness = size ** 2

        for x in range(size):
            for y in range(size):
                if self.fractal[x][y] == origin[x][y]:
                    self.fitness -= 1

        self.calculate_fitness_rotate(size, origin, cv2.ROTATE_90_CLOCKWISE)
        self.calculate_fitness_rotate(size, origin, cv2.ROTATE_180)
        self.calculate_fitness_rotate(size, origin, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def calculate_fitness_rotate(self, size, origin, rotation):
        tmpFitness = size ** 2
        fractal_rotated = cv2.rotate(self.fractal, rotation)
        for x in range(size):
            for y in range(size):
                if fractal_rotated[x][y] == origin[x][y]:
                    tmpFitness -= 1

        if tmpFitness < self.fitness:
            self.fitness = tmpFitness
            self.fractal = fractal_rotated

    def calculate_better_fitness(self, size, origin):
        self.calculate_better_fitness_general(size, origin, self.fractal)
        self.calculate_better_fitness_general(size, origin, cv2.rotate(self.fractal, cv2.ROTATE_90_CLOCKWISE))
        self.calculate_better_fitness_general(size, origin, cv2.rotate(self.fractal, cv2.ROTATE_180))
        self.calculate_better_fitness_general(size, origin, cv2.rotate(self.fractal, cv2.ROTATE_90_COUNTERCLOCKWISE))

    def calculate_better_fitness_general(self, size, origin, generated_fractal):
        points_not_drawn = 0
        points_not_needed = 0
        points_in_image = 0
        points_in_attractor = 0
        for x in range(size):
            for y in range(size):
                if origin[x][y] == 0:
                    points_in_image += 1
                    if generated_fractal[x][y] != 0:
                        points_not_drawn += 1

                if generated_fractal[x][y] == 0:
                    points_in_attractor += 1
                    if origin[x][y] != 0:
                        points_not_needed += 1

        attractor_relative_coverage = points_not_drawn / points_in_image
        points_outside_image = points_not_needed / points_in_attractor

        tmp_fitness = int((attractor_relative_coverage + points_outside_image) * 1000)
        if tmp_fitness < self.fitness:
            self.fitness = tmp_fitness
            self.fractal = generated_fractal

    def mutate(self):
        how_many_singels_to_mutate = random.randint(1, len(self.singels))
        for i in range(how_many_singels_to_mutate):
            singel_to_mutate = self.singels[random.randint(0, len(self.singels)-1)]
            singel_to_mutate.mutate()

    def __repr__(self):
        return f"Degree: {self.degree}, Fitness: {self.fitness}"
