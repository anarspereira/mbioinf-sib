from scipy import stats

def f_classification(dataset):
    '''
    Método que realiza o teste one-way ANOVA nos dados do dataset devolvendo os valores de F e p (em arrays)
    para cada feature
    :param dataset: Dataset, input dataset
    :return: F : np.array, F scores
            p : np.array, p-values
    '''
    classes = dataset.get_classes()
    groups = [dataset.X[dataset.y == c] for c in classes]
    F, p = stats.f_oneway(=groups)
    return F,p