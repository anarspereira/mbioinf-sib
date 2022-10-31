import numpy as np
from data.dataset import Dataset
from statistics.f_classification import f_classification

class SelectPercentile():
    '''
    Classe que integra os métodos responsáveis pela seleção de um percentil de features segundo a análise da
    variânica(score_func), sendo que o percentil a selecionar é dado pelo utilizador
    '''
    def __init__(self, score_func, percentile: int):
        '''
        Método que inicializa
        :param score_func: função de análise da variância
        :param k: número de features a selecionar
        '''
        self.score_func = score_func
        self.percentile = percentile #se em percentagem temos de depois dividir por 100
        self.F = None #ainda não os temos
        self.p = None

    def fit(self, dataset) -> self:
        '''
        Método que recebe o dataset e estima os valores de F e p para cada feature, usando a função score_func
        :param dataset: Dataset, input dataset
        :return: self
        '''
        F, p = self.score_func(dataset) #devolve os valores de F e p
        return self

    def transform(self, dataset):
        '''
        Método que seleciona as features com o valor de F mais alto no percentil pretendido
        :param dataset: Dataset, input dataset
        :return: Dataset, dataset com as features selecionadas
        '''
        value = len(dataset.features) #valor total de features no dataset
        mask = value * (percentile/100) #percentil dado em percentagem
        idxs = np.argsort(self.F)[- mask:]
        features = np.array(dataset.features)[idxs]  # vai selecionar as features utilizando os indxs
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)


    def fit_transform(self, dataset):
        '''
        Método que executa o método fit e depois o método transform
        :param dataset: input dataset
        :return: dataset com as variáveis selecionadas
        '''
        self.fit(dataset)
        return self.transform(dataset)

    if __name__ == '__main__':
        dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                      [0, 1, 4, 3],
                                      [0, 1, 1, 3]]),
                          y=np.array([0, 1, 0]),
                          features=["f1", "f2", "f3", "f4"],
                          label="y")
        select = SelectPercentile(f_classification,25)
        # chamar o método f_classification p/ o cálculo e introduzir valor de k
        select = select.fit(dataset)
        dataset = select.transform(dataset)
        print(dataset)