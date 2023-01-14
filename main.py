from evolutionary_algortihm import EvolutionaryAlgorithm

if __name__ == '__main__':
    # coefficients in file are in format a b c d e f
    data_file_name = "data/fern/fern.txt"
    img_file_name = "data/fern/fern400-5000.png"

    ea = EvolutionaryAlgorithm(data_file_name, img_file_name)
    ea.evolve()
