import numpy as np
from si.data.dataset_module import Dataset
from si.statistics.f_classification import f_classification
from si.io_package.csv_file import read_csv


class SelectPercentile:
    """
    Classe que integra os métodos responsáveis pela seleção de um percentil de features segundo a análise da
    variância(score_func), sendo que o percentil a selecionar é dado pelo utilizador.

    Parameters
    ----------
    score_func: callable
        Function taking dataset and returning a pair of arrays (scores, p_values).
    percentile: int
        Percentile value.

    Attributes
    ----------
    F: array, shape (n_features,)
        F scores of features
    p: array, shape (n_features,)
        p-values of F-scores
    """

    def __init__(self, score_func, percentile: int):
        """
        Construtor

        :param score_func: função de análise da variância (f_classification() ou f_regression())
        :param percentile: valor do percentile. apenas F-scores acima desse valor permanecem no dataset filtrado
        """
        self.score_func = score_func
        self.percentile = percentile # se em percentagem temos de depois dividir por 100
        self.F = None # ainda não os temos
        self.p = None

    def fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        Método que recebe o dataset e estima os valores de F e p para cada feature, usando a função score_func

        :param dataset: Dataset, input dataset

        :return: self
        """
        self.F, self.p = self.score_func(dataset) # devolve os valores de F e p
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Método que seleciona as features com o valor de F mais alto no percentil pretendido

        :param dataset: Dataset, input dataset

        :return: Dataset, dataset com as features selecionadas
        """
        features_percentile = int(len(dataset.features_names) * self.percentile / 100)  # calcula o nº de features
        # selecionadas com F score mais alto até ao valor do percentile indicado (50% de 10 features equivale a 5
        # features)
        idxs = np.argsort(self.F)[-features_percentile:]
        best_features = dataset.X[:, idxs]
        best_features_names = [dataset.features_names[i] for i in idxs]

        return Dataset(best_features, dataset.y, best_features_names, dataset.label_names)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Método que executa o método fit e depois o método transform

        :param dataset: input dataset

        :return: dataset com as variáveis selecionadas
        """
        self.fit(dataset)
        return self.transform(dataset)

if __name__ == '__main__':
    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features_names=["f1", "f2", "f3", "f4"],
                      label_names="y")
    # dataset = read_csv('C:/Users/Ana/Documents/GitHub/mbioinf-sib/datasets/cpu.csv', sep=",", label=True)
    select = SelectPercentile(f_classification, percentile=25) # chamar o método f_classification p/ o cálculo e introduzir valor de percentile
    select = select.fit(dataset)
    dataset = select.transform(dataset)
    print(dataset)
    print(dataset.features_names)