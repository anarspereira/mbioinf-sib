import numpy as np


def read_data_file(filename: str, sep: str, label: bool) -> object:
    """
    Método para ler csv e retornar um dataset.

    :param filename: Nome ou caminho do ficheiro
    :param sep: Qual o separador entre valores (default: ',')
    :param label: Booleano

    :return: Dataset.
    """
    return np.genfromtxt(filename, **kwargs)

def write_csv(filename: str, dataset: object, sep: str, label: bool) -> object:
    """
    Método para escrever csv.

    :param filename: Nome ou caminho do ficheiro
    :param dataset:
    :param sep: Qual o separador entre valores (default: ',')
    :param label:

    :return:
    """
    return np.savetxt(filename, dataset, sep, label)