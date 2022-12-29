import pandas as pd
from si.data.dataset import Dataset

def read_csv(filename: str, sep: str = ",", features: bool = False, label: bool = False) -> pd.DataFrame:

    # sep: str, features: bool, label: bool

    """
    Ler ficheiros csv.

    :param filename: Nome ou caminho do ficheiro csv
    :param sep: Qual o separador entre valores (default: ',')
    :param features: Booleano: True se tem o nome das features; false se não tem o nome das features
    :param label: Booleano: True se tem labels; false se não tem labels

    :return: Pandas dataframe do csv
    """
    data = pd.read_csv(filename, delimiter=sep)
    if features and label:  # se temos as features e labels
        features = data.colums[:-1]
        x = data.iloc[1:, 0:, -1].to_numpy()  # queremos começar na primeira linha #passa para numpy
        y = data.iloc[:, -1].to_numpy()
        features = None
        label = None
    elif features and not label:  # se temos as features, mas não as labels
        features = data.columns
        y = None
    elif not features and label:  # se não temos as fetaures, mas temos as labels
        label = data.columns[-1]
        y = data.iloc[:, -1]
        data = data.iloc[:, :-1]
    else:  # quando não temos nem as features nem as labels
        y = None
    return Dataset(data, y, features, label)

def write_csv_file(filename: str, dataset: Dataset, sep: str = ",", features: bool = False, label: bool = None) -> None:
    # TODO: não está a assumir pd.Dataframe
    # TODO: completar documentação
    # TODO: resolver método
    """
    Escrever ficheiros csv.

    :param filename: Nome ou caminho do ficheiro csv
    :param dataset: Dataset a ser escrito
    :param sep: Qual o separador entre valores (default: ',')
    :param features:
    :param label:

    :return: Pandas dataframe do csv
    """
    data = pd.DataFrame(dataset.x)  # construir de novo o dataframe e para isso basta passar o x
    if features:
        data.colums = dataset.features
    if label:
        data[dataset.label] = dataset.y

    data.to_csv(filename, sep=sep, index=False)  # passa os dados do dataset escrito para um ficheito csv


if __name__ == '__main__':
    dsiris = pathlib.Path("mbioinf-sib/datasets/iris.csv") # usar os.path() ou pathlib.Path()
    print(read_csv(dsiris))