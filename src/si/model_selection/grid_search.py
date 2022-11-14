import itertools
from typing import Dict, Callable, List, Tuple

from si.data.dataset import Dataset
from si.model_selection.cross_validate import cross_validate


def grid_search_cv (model,
                    dataset: Dataset,
                    parameter_grid: Dict[str, Tuple],
                    scoring: Callable = None,
                    cv: int = 3,
                    test_size: float = 0.2) -> Dict[str, List[float]]:
    # exemplo parameter_grid:
    # {alpha: [0.001, 0.00001, ....],
    # max_iter: [1000, 2000, ....],
    # l2_penalty: [1, 2, 3, 4, ....]}
    # vai verificar se de fact os parâmetros existem no modelo - existe a função has_attribute ou hasattr
    # vai gerar uma lista em que combina os parâmetros uns com os outros e depois faz um set (uma atribuição) dos parâmetros

    """

    :param model:
    :param dataset:
    :param parameter_grid:
    :param scoring:
    :param cv:
    :param test_size:
    :return:
    """

    # validate the parameter grid
    # para cada parâmetro que o utilizador deu no dicionário, verifico se o modelo que ele deu tem o parâmetro
    for parameter in parameter_grid:
        if not hasattr(model, parameter): # se não tiver, faz um raise do erro
            raise AttributeError(f"Model {model} does not have parameter {parameter}.")

    scores = []

    # for each combination, vamos buscar todas as combinações
    for combination in itertools.product(*parameter_grid.values()): # * faz unpack de todos os valores

        # parameter configuration, criando um dicionário intermédio para guardar o parameter e o value
        parameters = {}

        # set the parameters, vamos atribuir cada combinação ao modelo
        for parameter, value in zip(parameter_grid.keys(), combination): # o zip itera dois iterables (duas sequências) ao mesmo tempo e atribui ao parameter o primeiro e ao value o segundo
            setattr(model, parameter, value) # set attribute do objeto (modelo), o parâmetro (nome do atributo no modelo) e o seu valor
            parameters[parameter] = value

        # cross validate the model - treina o modelo e retorna o dicionário de scores
        score = cross_validate(model=model, dataset=dataset, scoring=scoring, cv=cv, test_size=test_size)

        # add the parameter configuration
        score['parameters'] = parameters

        # add the score
        scores.append(score)

    return scores

if __name__ == '__main__':
    # import dataset
    from si.linear_model.logistic_regression import LogisticRegression

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)

    # initialize the logistic regression model
    # ver o resto do código no github