import copy

import cv2
import random
import numpy as np

from utils.consts import *
from models.ifs import Ifs
from models.singel import Singel
from utils.math_helper import MathHelper


class EvolutionaryAlgorithm:
    # needed to create fractal from IFS
    iterations: int
    # needed to create fractal from IFS
    size: int

    # those min and max values concern only constant terms of IFS, as non constant terms have to be between -1 and 1
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
        self.read_fractal_from_file(img_file_name)
        self.setup()

    def read_data_from_file(self, file_name: str):
        """
            Reads base data about creating fractals from IFS and generating singels
        """
        f = open(file_name, "r")

        self.iterations = int(f.readline())
        self.size = int(f.readline())
        self.min_coefficient = float(f.readline())
        self.max_coefficient = float(f.readline())

        f.close()

    def read_fractal_from_file(self, file_name: str):
        """
            Reads fractal from file as ndarray and transforms it into binary representation
        """
        self.fractal = cv2.imread(file_name)
        self.fractal = cv2.cvtColor(self.fractal, cv2.COLOR_BGR2GRAY)
        self.fractal = cv2.threshold(self.fractal, 254, 255, cv2.THRESH_BINARY)[1]

    def setup(self):
        self.setup_degree_probabilities()
        self.setup_universe()
        self.setup_population()
        self.setup_individuals_in_species()

    def setup_degree_probabilities(self):
        """
            Sets up VD vector (probability of generating IFS with degree) with base values defined in utils/consts.py
        """
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

    def setup_individuals_in_species(self):
        """
            Sets up VC vector (how many IFS are there for every degree) based on current population
        """
        self.individuals_in_species = {}
        for degree in self.degree_probabilities:
            self.individuals_in_species[degree] = 0

        for ifs in self.population:
            self.individuals_in_species[ifs.degree] += 1

    def setup_universe(self):
        """
            Creates universe of singles from which IFS are created during whole algorithm
        """
        for i in range(UNIVERSE_SIZE):
            coefficients = []
            # a b c d
            for j in range(4):
                coefficients.append(random.uniform(-0.99999, 1))

            # e f
            for j in range(2):
                coefficients.append(random.uniform(self.min_coefficient, self.max_coefficient))

            probability = random.uniform(0, 1)

            self.universe.append(Singel(coefficients, probability))

    def setup_population(self):
        """
            Creates initial population
        """
        self.population.extend(self.self_creation(POPULATION_SIZE))

    def self_creation(self, size: int) -> list[Ifs]:
        """
            Creates given number of IFS from universe of singels
        """
        result = []
        for i in range(size):
            degree = random.choices(list(self.degree_probabilities.keys()), list(self.degree_probabilities.values()))[0]
            singels = []
            for j in range(degree):
                singels.append(random.choice(self.universe))

            result.append(Ifs(singels))

        return result

    def fitness_function(self):
        """
            Calculates fitness value for each IFS in current population
        """
        for ifs in self.population:
            ifs.calculate_fitness(self.fractal, self.iterations, self.size)

    def evolve(self):
        """
            Performs main evolutionary process
        """
        for gen in range(GENERATIONS):
            self.fitness_function()
            self.adapt_degree_probabilities()
            new_population = []

            best_individuals = self.get_best_individual_of_each_degree()
            for best in best_individuals:
                if best.fitness <= FINAL_THRESHOLD:
                    cv2.imwrite("data/fern/fern-result.png", best.fractal)
                    return

                if best.fitness <= TH_THRESHOLD:
                    new_population.append(copy.deepcopy(best))

            remaining = POPULATION_SIZE - len(new_population)

            # N1
            new_population.extend(self.create_ifs_by_crossover(int(remaining/4)))
            remaining -= int(remaining/4)

            # N2
            new_population.extend(self.self_creation(int(remaining / 3)))
            remaining -= int(remaining / 3)

            # N3
            new_population.extend(self.create_ifs_by_inter_species_crossover(int(remaining / 2)))
            remaining -= int(remaining / 2)

            # N4
            new_population.extend(self.create_ifs_by_reassortment(remaining))

            for ifs in new_population:
                rand = random.uniform(0, 1)
                if rand < MUTATION_PROBABILITY:
                    ifs.mutate()

            self.population = new_population
            self.setup_individuals_in_species()

    def adapt_degree_probabilities(self):
        """
            Changes VD vector based on fitness value of IFS in current population
        """
        best_individuals = self.get_best_individual_of_each_degree()
        best_individuals_fitness = [i.fitness for i in best_individuals]

        normalized_values = MathHelper.normalize_probabilities(best_individuals_fitness, True)

        for key in self.degree_probabilities:
            self.degree_probabilities[key] = 0

        for idx, item in enumerate(best_individuals):
            self.degree_probabilities[item.degree] = normalized_values[idx]

    def get_best_individual_of_each_degree(self) -> list[Ifs]:
        """
            Gets list of IFS with best fitness from current population, one for each degree
        """
        best_individuals = []
        for degree in self.degree_probabilities:
            if self.individuals_in_species[degree] > 0:
                ifs_for_degree = [x for x in self.population if x.degree == degree]
                best_individuals.append(sorted(ifs_for_degree, key=lambda x: x.fitness)[0])

        return best_individuals

    # TODO
    def create_ifs_by_crossover(self, how_many: int) -> list[Ifs]:
        raise NotImplementedError()

    # TODO
    def arithmetic_crossover(self):
        raise NotImplementedError()

    # TODO
    def vector_crossover(self):
        raise NotImplementedError()

    # TODO
    def create_ifs_by_inter_species_crossover(self, how_many: int) -> list[Ifs]:
        raise NotImplementedError()

    # TODO
    def inter_species_crossover(self):
        raise NotImplementedError()

    # TODO
    def create_ifs_by_reassortment(self, how_many: int) -> list[Ifs]:
        raise NotImplementedError()

    # TODO
    def reassortment(self):
        raise NotImplementedError()

    # TODO
    def purge_species(self):
        raise NotImplementedError()
