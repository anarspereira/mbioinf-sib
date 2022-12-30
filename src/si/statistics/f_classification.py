from typing import Tuple, Union

import numpy as np
from scipy import stats

from si.data.dataset_module import Dataset


def f_classification(dataset: Dataset) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[float, float]]:
    """
    Método que realiza o teste one-way ANOVA nos dados do dataset devolvendo os valores de F e p (em arrays)
    para cada feature.
    O F-value permite analisar se a média entre dois ou mais grupos de fatores são significativamente diferentes.
    As samples são agrupadas pelas labels do dataset.

    :param dataset: Labeled dataset

    :return: F: np.array, F scores
             p: np.array, p-values
    """
    classes = dataset.get_classes()
    groups = [dataset.X[dataset.y == c] for c in classes]
    F, p = stats.f_oneway(*groups)
    return F,p