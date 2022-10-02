import os.path
import pathlib

import pandas as pd


def read_csv(filename: str, **kwargs) -> pd.DataFrame:
    #TODO: não está a assumir pd.Dataframe
    #TODO: completar documentação
    #TODO: resolver método

    # sep: str, features: bool, label: bool

    """

    :param filename: Nome ou caminho do ficheiro
    :param sep: Qual o separador entre valores (default: ',')
    :param features:
    :param label:

    :return: Pandas dataframe do CSV
    """
    if not os.path.exists(filename):
        raise Exception('Nome do ficheiro ou path inválidos.')
    else:
        return pd.read_csv(filename, **kwargs)

def write_csv(filename: str, dataset: object, sep: str, features: bool, label: bool) -> pd.DataFrame:
    # TODO: não está a assumir pd.Dataframe
    # TODO: completar documentação
    # TODO: resolver método
    """

    :param filename: Nome ou caminho do ficheiro
    :param dataset:
    :param sep: Qual o separador entre valores (default: ',')
    :param features:
    :param label:

    :return:
    """
    if not os.path.exists(filename):
        raise Exception('Nome do ficheiro ou path inválidos.')
    else:
        return pd.write_csv(filename, dataset)


if __name__ == '__main__':
    dsiris = pathlib.Path("mbioinf-sib/datasets/iris.csv") # usar os.path() ou pathlib.Path()
    print(read_csv(dsiris))