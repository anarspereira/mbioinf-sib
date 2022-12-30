from typing import Callable

import numpy as np

from si.data.dataset_module import Dataset
from si.statistics.f_classification import f_classification


class SelectKBest:
    """
    Classe que integra os métodos responsáveis pela seleção de k melhores features segundo a análise da variância
    (score_func), sendo k um valor dado pelo utilizador.

    Scoring function:
    - f_classification: ANOVA F-value entre label/feature para problemas de classificação.
    - f_regression: F-value obtained from F-value of r's pearson correlation coefficients para problemas de regressão.

    Parameters
    ----------
    score_func: callable
        Function taking dataset and returning a pair of arrays (scores, p_values).
    k: int, default=10
        Number of top features to select.

    Attributes
    ----------
    F: array, shape (n_features,)
        F scores of features
    p: array, shape (n_features,)
        p-values of F-scores
    """

    def __init__(self, score_func: Callable = f_classification, k: int = 10):
        """
        Construtor

        :param score_func: função de análise da variância (recebe o dataset e retorna um par de arrays (scores, p-values)
        :param k: número de features a selecionar (default: 10)
        """
        self.score_func = score_func
        self.k = k
        self.F = None # ainda não os temos
        self.p = None

    def fit(self, dataset: Dataset) -> 'SelectKBest':
        """
        Método que recebe o dataset e estima os valores de F-score e p-value para cada feature, usando a função score_func.

        :param dataset: Dataset, input dataset

        :return: self
        """
        self.F, self.p = self.score_func(dataset) # devolve os valores de F e p
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Método que seleciona k features com os scores mais altos (calculados através do método fit)

        :param dataset: Dataset, input dataset

        :return: dataset : Dataset, um dataset com as features com scores mais altos
        """
        idxs = np.argsort(self.F)[- self.k:] # argsort devolve os índices de ordenação do array
        features = np.array(dataset.features_names)[idxs] # vai selecionar as features utilizando os indxs
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features_names=list(features), label_names=dataset.label_names)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Método que executa o método fit e depois o método transform - faz o fit do SelectKBest e transforma o dataset
        selecionando as com k-highest scoring features.

        :param dataset: Dataset, input dataset

        :return: Dataset com as k-highest scoring features.
        """
        self.fit(dataset)
        return self.transform(dataset)

if __name__ == '__main__':
    dataset = Dataset(X = np.array([[0, 2, 0, 3],
                                    [0, 1, 4, 3],
                                    [0, 1, 1, 3]]),
                      y = np.array([0, 1, 0]),
                      features_names=["f1", "f2", "f3", "f4"],
                      label_names="y")
    select = SelectKBest(f_classification, 1) # chamar o método f_classification para o cálculo e introduzir valor de k
    new_dataset = select.fit_transform(dataset)
    print(new_dataset.X)
    print(new_dataset.features_names)