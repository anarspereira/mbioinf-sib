from typing import Dict, Callable, List

import numpy as np

from si.data.dataset_module import Dataset
from si.model_selection.split import train_test_split


def cross_validate (model, dataset: Dataset, scoring: Callable = None, cv: int = 3, test_size: float = 0.2) -> Dict[str, List[float]]:
    """
    Método para fazer a cross validation dos dataset de treino e teste.

    :param model: Modelo a validar
    :param dataset: Dataset de validação
    :param scoring: Função de score
    :param cv: Número de folds
    :param test_size: Tamanho do dataset de teste

    :return: Dicionário com scores dos modelos do dataset.
    """

    scores = {
        'seeds': [],
        'train': [],
        'test': []
    }

    # for each fold
    for i in range(cv): # para cada cross validation
        # get random seed
        random_state = np.random.randint(0, 1000)

        # store seed
        scores['seeds'].append(random_state)

        # split the dataset in train and test
        train, test = train_test_split(dataset=dataset, test_size=test_size, random_state=random_state)

        # fit the model on the train set
        model.fit(train)

        # score the model on the test set
        # o utilizador pode querer usar a scoring function que está no nosso modelo
        if scoring is None:

            # store the train score
            scores['train'].append(model.score(train))

            # store the test score
            scores['test'].append(model.score(test))

        else:
            # vamos fazer o nosso próprio score - ctrl+C, ctrl+V do score do logistic regression ou knn
            y_train = train.y
            y_test = test.y

            # store the train score
            scores['train'].append(scoring(y_train, model.predict(train)))

            # store the test score
            scores['test'].append(scoring(y_test, model.predict(test)))

    return scores

# o output vai ser algo do género:
# seeds: [10, 9, 7]
# train: [0.9, 0.91, 0.87]
# test: [0.85, 0.86, 0.4]