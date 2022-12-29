import pathlib
import sys
sys.path.insert(0, 'src/si')

from si.data.dataset import Dataset

import numpy as np

def read_data_file (filename: str, sep: str=",", label: bool=False):
    """
    Método para ler csv e retornar um dataset.

    :param filename: Nome ou caminho do ficheiro csv
    :param sep: Qual o separador entre valores (default: ',')
    :param label: Booleano: True se tem labels; false se não tem labels

    :return: Dataset.
    """
    data = np.genfromtxt(filename, delimiter=sep)

    if label is True:  # se o ficheiro tem label definida
        x = data[:, :-1]  # todas as colunas, exceto a última
        y = data[:, -1]  # todas as linhas da última coluna
    else:  # se o ficheito não tem label definida
        x = data
        y = None  # não tem labels

    return Dataset(x, y)  # cria dataset com os dados

def write_data_file (filename: str, sep: str = ",", label: bool = False):
    """
    Método para escrever csv.

    :param filename: Nome ou caminho do ficheiro csv
    :param sep: Qual o separador entre valores (default: ',')
    :param label: Booleano: True se tem labels; false se não tem labels
    """
    if not sep:
        sep = " "

    if label is True:
        data = np.hstack(Dataset.X, Dataset.y.reshape(-1, 1)) # reune os arrays na horizontal
    else:
        data = Dataset.X

    return np.savetxt(filename, data, delimiter=sep) # guarda o array num ficheiro

if __name__ == '__main__':
    # file = pathlib.Path('mbioinf-sib/datasets/breast-bin.data')
    # test = read_data_file(file, label=4)
    # write_data_file(test, "write_data_file_test.csv", label=4)
    pass