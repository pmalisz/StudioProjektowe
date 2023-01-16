import math
import numpy as np


class MathHelper:
    @staticmethod
    def normalize_probabilities(probabilities: list[float], invert: bool = False) -> list[float]:
        normalized_probabilities = [round(float(i) / sum(probabilities), 2) for i in probabilities]

        if invert:
            sum_div = [sum(probabilities) / float(i) for i in probabilities]
            normalized_probabilities = [round(float(i) / sum(sum_div), 2) for i in sum_div]

        diff = np.sum(normalized_probabilities) - 1
        if not math.isclose(diff, 0):
            while True:
                random_index = np.random.randint(0, len(normalized_probabilities))
                if normalized_probabilities[random_index] > diff:
                    normalized_probabilities[random_index] -= diff
                    break

        return normalized_probabilities
