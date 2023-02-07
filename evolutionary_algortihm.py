import copy
from fractions import Fraction
from typing import Callable

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
                coefficients.append(random.uniform(-0.99, 1))

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
            degree = self.get_random_degree_based_on_vd()
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
                if gen in (249, 499, 749, 999):
                    cv2.imwrite(f"data/IFS tree/better-tree-gen{gen+1}-degree{best.degree}-result.png", best.fractal)
                    f = open("data/IFS tree/ifsTree-fitness.txt", "a")
                    f.write(f"gen: {gen+1}, degree: {best.degree}, fitness: {best.fitness}\n")
                    f.close()

                if best.fitness <= FINAL_THRESHOLD:
                    cv2.imwrite("data/IFS tree/better-tree-result.png", best.fractal)
                    return

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
            #self.purge_species()

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
            Gets list of IFS with the best fitness from current population, one for each degree
        """
        best_individuals = []
        for degree in self.degree_probabilities:
            if self.individuals_in_species[degree] > 0:
                ifs_for_degree = [x for x in self.population if x.degree == degree]
                best_individuals.append(sorted(ifs_for_degree, key=lambda x: x.fitness)[0])

        return best_individuals

    def create_ifs_by_crossover(self, how_many: int) -> list[Ifs]:
        """
            Performs vector or arithmetic crossover operations with equal probability
        """
        result = []
        for i in range(how_many//2):
            degree = self.get_random_degree_based_on_vd()
            ifs_for_degree = [x for x in self.population if x.degree == degree]

            # Since we are minimizing fitness, we want lesser value to be more probable
            weights = [1/x.fitness for x in ifs_for_degree]

            random_individuals = random.choices(ifs_for_degree, weights, k=2)
            parent1 = random_individuals[0]
            parent2 = random_individuals[1]

            result.extend(self.arithmetic_crossover(parent1, parent2))

            # random_crossover = random.getrandbits(1)
            # if random_crossover == 0:
            #     result.extend(self.arithmetic_crossover(parent1, parent2))
            # else:
            #     result.extend(self.vector_crossover(parent1, parent2))

        return result

    def arithmetic_crossover(self, parent1: Ifs, parent2: Ifs) -> tuple[Ifs, Ifs]:
        point_of_division = random.randint(0, parent1.degree-1)
        new_singels1 = parent1.singels[:point_of_division] + parent2.singels[point_of_division:]
        new_singels2 = parent2.singels[:point_of_division] + parent1.singels[point_of_division:]

        return Ifs(new_singels1), Ifs(new_singels2)

    # TODO
    def vector_crossover(self, parent1: Ifs, parent2: Ifs) -> tuple[Ifs, Ifs]:
        raise NotImplementedError()

    def choose_parents_and_perform_crossover(self, how_many: int, crossover: Callable) -> list[Ifs]:
        """
            wrapper for crossover operations
        """
        children = []
        possible_parents = random.sample(self.population, len(self.population))
        for _ in range(how_many):
            # basically do while loop
            while True:
                parent1 = possible_parents.pop()
                parent2 = possible_parents.pop()
                if parent1 != parent2 and parent2.degree != parent1.degree:
                    break

                possible_parents.append(parent1)
                possible_parents.append(parent2)
                random.shuffle(possible_parents)
            children.extend(crossover(parent1, parent2))

        return children

    def create_ifs_by_inter_species_crossover(self, how_many: int) -> list[Ifs]:
        """
            Performs inter_species_crossover how_many//2 times
        """
        return self.choose_parents_and_perform_crossover(how_many//2, self.inter_species_crossover)

    def inter_species_crossover(self, parent1: Ifs, parent2: Ifs) -> tuple[Ifs, Ifs]:
        child1_singels = []
        child2_singels = []

        for _ in range(parent1.degree):
            singel1 = random.choice(parent1.singels)
            singel2 = random.choice(parent2.singels)
            crossed_singel = Singel.cross_singels(singel1, singel2)
            child1_singels.append(crossed_singel)

        for _ in range(parent2.degree):
            singel1 = random.choice(parent1.singels)
            singel2 = random.choice(parent2.singels)
            crossed_singel = Singel.cross_singels(singel1, singel2)
            child2_singels.append(crossed_singel)

        return Ifs(child1_singels), Ifs(child2_singels)

    def create_ifs_by_reassortment(self, how_many: int) -> list[Ifs]:
        """
            Performs reassortment how_many//2 times
        """
        return self.choose_parents_and_perform_crossover(how_many//2, self.reassortment)

    def reassortment(self, parent1: Ifs, parent2: Ifs) -> tuple[Ifs, Ifs]:
        """
            Creates two offspring by exchanging singels between parents
        """
        def choose_parent_with_singels(parents: list[list[Singel]], child_singels: list[Singel]):
            """
                Wrapper for preventing choosing list with no singels
            """
            random.shuffle(parents)
            selected_parent = parents.pop()
            if len(selected_parent) == 0:
                selected_parent = parents.pop()
            random.shuffle(selected_parent)
            child_singels.append(selected_parent.pop())

        parent1_singels = [singel for singel in parent1.singels]
        parent2_singels = [singel for singel in parent2.singels]
        child1_singels = []
        child2_singels = []

        for _ in range(parent1.degree):
            choose_parent_with_singels([parent1_singels, parent2_singels], child1_singels)
        for _ in range(parent2.degree):
            choose_parent_with_singels([parent1_singels, parent2_singels], child2_singels)

        return Ifs(child1_singels), Ifs(child2_singels)

    def get_random_degree_based_on_vd(self):
        return random.choices(list(self.degree_probabilities.keys()), list(self.degree_probabilities.values()))[0]

    # def purge_species(self):
    #     """
    #     Removes whole species if it's population is less than %5 or it's average fitness is lower than THRESHOLD
    #     and creates new specie if it's possible
    #     """
    #     population_size = len(self.population)
    #     best_indivs = self.get_best_individual_of_each_degree()
    #     for ifs in best_indivs:
    #         if self.individuals_in_species[ifs.degree] / population_size < 0.05: #or self.best_fitness_for_specie[species] < #THRESHOLD:
    #             for indiv in self.population:
    #                 if indiv.degree == ifs.degree:
    #                     self.population.remove(indiv)
    #
    #             self.individuals_in_species[ifs.degree] = 0
    #             if self.get_next_free_degree() != -1:
    #                 self.create_new_specie()
    #                 pass
    #                 #self.individuals_in_species[self.get_next_free_degree()] = 1 #have to generate new specie
    #
    # def get_next_free_degree(self) -> int:
    #     """
    #     Returns first free degree on the right of specie with best fitness
    #     """
    #     specie_with_best_fitness = sorted(self.get_best_individual_of_each_degree(), key=lambda x: x.fitness)[0].degree
    #     for i in range(specie_with_best_fitness + 1, MAX_DEGREE + 1):
    #         if self.individuals_in_species[i] == 0:
    #             return i
    #     else:
    #         return -1
    #
    # def create_new_specie(self):
    #     raise NotImplementedError()
