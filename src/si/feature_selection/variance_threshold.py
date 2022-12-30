import numpy as np
from si.data.dataset_module import Dataset

class VarianceThreshold:
    """
    Classe que integra os métodos de análise e remoção dos features com um valor de variância abaixo do
    valor de threshold
    """

    def __init__(self, threshold: float):
        """
        Construtor

        :param threshold: float, valor de corte
        """
        if threshold < 0:
            raise ValueError("Threshold must be non-negative.")

        self.threshold = threshold
        self.variance = None # para já está vazio porque nós não temos o dataset aqui

    def fit(self, dataset: Dataset) -> 'VarianceThreshold':
        """
        Método que recebe o dataset e estima o atributo a partir dos dados, sendo que neste caso calcula a variância
        para cada coluna/feature

        :param dataset: Dataset, input dataset

        :return: self
        """
        variance = np.var(dataset.X) # ou podíamos usar o método da aula passada
        self.variance = variance # atribuir a variável
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Método que seleciona todas as features com variância (calculada através do método fit) superiores ao valor
        de threshold (valor de corte dado pelo utilizador)

        :param dataset: input dataset

        :return: dataset com as variáveis selecionadas
        """
        features_mask = self.variance > self.threshold # mask é valor bool para identificar as colunas que estão acima
        X = dataset.X[:, features_mask]
        features = np.array(dataset.features_names)[features_mask]

        return Dataset(X=X, y=dataset.y, features_names=list(features), label_names=dataset.label_names)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Método que executa o método fit e depois o método transform

        :param dataset: input dataset

        :return: dataset com as variáveis selecionadas
        """
        self.fit(dataset)
        return self.transform(dataset)

if __name__ == '__main__':
    dataset = Dataset(X=np.array([[0,2,0,3],
                                    [0,1,4,3],
                                    [0,1,1,3]]),
                      y=np.array([0, 1, 0]),
                      features_names=["f1", "f2", "f3", "f4"],
                      label_names="y")
    selector = VarianceThreshold(0) # dar o valor de corte que queremos considerar
    selector = selector.fit(dataset)
    dataset = selector.transform(dataset)
    print(dataset.features_names)