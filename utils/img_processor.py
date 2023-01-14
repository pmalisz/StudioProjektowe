import numpy as np
import matplotlib.pyplot as plt


def draw_function(probability):  # number of position in list define number of function
    return np.random.choice(len(probability), p=probability)


def apply_function(function, point):
    matrix_to_multiplicity = np.array([point[0], point[1], 1]).astype(np.float)
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


class ImgProcessor:
    @staticmethod
    def generate_fractal(iterations, size, functions, probability, draw_img=False):
        points = [[0, 0]]
        for i in range(iterations):
            function_number = draw_function(probability)
            k = apply_function(functions[function_number], points[-1])
            if k[0] is not float("inf") and k[1] is not float("inf"):
                points.append(k)
        points = scale_and_round(points, size)
        if draw_img:
            draw_image(points, size)
        return points
