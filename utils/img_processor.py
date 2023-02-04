import math

import numpy as np


def draw_function(probability):  # number of position in list define number of function
    return np.random.choice(len(probability), p=probability)


def apply_function(function, point):
    matrix_to_multiplicity = np.array([point[0], point[1], 1])
    x = np.sum(np.multiply(function[0], matrix_to_multiplicity))
    y = np.sum(np.multiply(function[1], matrix_to_multiplicity))
    return [x, y]


def scale_and_round(points, size):
    normalizedData = np.round((points - np.min(points)) / (np.max(points) - np.min(points)) * size)
    return normalizedData


def get_image(points, size):
    points = points.astype(int)
    image = np.full((size + 1, size + 1), 255)
    for point in points:
        if point[0] < 0 or point[1] < 0:
            return image
        image[point[0]][point[1]] = 0

    return image


class ImgProcessor:
    @staticmethod
    def generate_fractal(iterations, size, functions, probability):
        points = [[0, 0]]
        for i in range(iterations):
            function_number = draw_function(probability)
            k = apply_function(functions[function_number], points[-1])
            if k[0] is not float("inf") and k[1] is not float("inf"):
                points.append(k)

        points = scale_and_round(points, size)

        return get_image(points, size)
