from evolutionary_algortihm import EvolutionaryAlgorithm

if __name__ == '__main__':
    """
        coefficients in file are in format a b c d e f
    """

    data_file_name = "data/IFS tree/ifsTree.txt"
    img_file_name = "data/IFS tree/ifsTree.png"

    ea = EvolutionaryAlgorithm(data_file_name, img_file_name)
    ea.evolve()
