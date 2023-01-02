import random

import cv2
import numpy as np

from utils.consts import *
from models.ifs import Ifs
from models.singel import Singel


class EvolutionaryAlgorithm:
    iterations: int
    size: int
    min_coefficient: float
    max_coefficient: float
    fractal: np.ndarray

    universe: list[Singel]
    population: list[Ifs]
    # VD
    degree_probabilities: dict[int, float]
    # VC
    individuals_in_species: dict[int, int]

    def __init__(self, data_file_name: str, img_file_name: str):
        self.universe = []
        self.population = []
        self.read_data_from_file(data_file_name)
        self.fractal = cv2.imread(img_file_name)
        self.setup()

    def read_data_from_file(self, file_name: str):
        f = open(file_name, "r")

        self.iterations = int(f.readline())
        self.size = int(f.readline())
        self.min_coefficient = float(f.readline())
        self.max_coefficient = float(f.readline())

        f.close()

    def setup(self):
        self.setup_degree_probabilities()
        self.setup_universe()
        self.setup_population()

    def setup_degree_probabilities(self):
        self.degree_probabilities = BASE_DEGREE_PROBABILITIES

        if MIN_DEGREE >= MAX_DEGREE:
            self.degree_probabilities[MIN_DEGREE + 1] = REMAINING_DEGREE_PROBABILITY
        else:
            remaining_probability = REMAINING_DEGREE_PROBABILITY
            for i in range(MIN_DEGREE + 1, MAX_DEGREE):
                remaining_degrees = MAX_DEGREE - i + 1
                probability = round(remaining_probability * (1 / remaining_degrees + 1 / i), 2)
                self.degree_probabilities[i] = probability
                remaining_probability -= probability

            self.degree_probabilities[MAX_DEGREE] = round(remaining_probability, 2)

    def setup_universe(self):
        for i in range(UNIVERSE_SIZE):
            coefficients = []
            for j in range(6):
                coefficients.append(random.uniform(self.min_coefficient, self.max_coefficient))

            self.universe.append(Singel(coefficients))

    def setup_population(self):
        self.population.extend(self.self_creation(POPULATION_SIZE))
        self.fitness_function()

    def self_creation(self, size: int) -> list[Ifs]:
        result = []
        for i in range(size):
            degree = random.choices(list(self.degree_probabilities.keys()), list(self.degree_probabilities.values()))[0]
            singels = []
            for j in range(degree):
                singels.append(random.choice(self.universe))

            result.append(Ifs(singels))

        return result

    # TODO
    def fitness_function(self):
        raise NotImplementedError()

    # TODO
    def evolve(self):
        raise NotImplementedError()

    # TODO
    def get_best_individual_for_each_degree(self) -> list[Ifs]:
        raise NotImplementedError()

    # TODO
    def arithmetic_crossover(self):
        raise NotImplementedError()

    # TODO
    def vector_crossover(self):
        raise NotImplementedError()

    # TODO
    def inter_species_crossover(self):
        raise NotImplementedError()

    # TODO
    def reassortment(self):
        raise NotImplementedError()

    # TODO
    def mutate(self):
        raise NotImplementedError()

    # TODO
    def purge_species(self):
        raise NotImplementedError()