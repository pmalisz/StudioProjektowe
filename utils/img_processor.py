import numpy as np
import matplotlib.pyplot as plt


def generate_fractal(self, iterations, size, functions, probability):
    points = [[0, 0]]
    for i in range(iterations):
        function_number = draw_function(probability)
        k = apply_function(functions[function_number], points[-1])
        points.append(k)
    points = scale_and_round(points, size)
    draw_image(points, size)
    return points


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


def draw_image(points, size):
    points = points.astype(int)
    image = np.zeros((size + 1, size + 1))
    for point in points:
        image[point[0]][point[1]] = 255
    plt.imshow(image)
