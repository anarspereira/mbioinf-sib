import numpy as np
from matplotlib import pyplot as plt

from si.data.dataset_module import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.sigmoid_function import sigmoid_function


class LogisticRegression:
    """
    The LogisticRegression is a linear model using the L2 regularization and the sigmoid function.
    This model uses the linear regression problem using an adapter Gradient Descent technique.

    Parameters
    ----------
    l2_penalty: float
        Coeficiente da regularização L2
    alpha: float
        Taxa de aprendizagem (learning rate)
    max_iter: int
        Número máximo de iterações

    Attributes
    ----------
    theta: np.array
        Coeficientes/parâmetros do modelo linear para as variáveis de entrada (features)
        Por exemplo, x0 * theta[0] + x1 * theta[1] + ...
    theta_zero: float
        Coeficiente/parâmetro 0, conhecido por interceção do modelo linear.
        Por exemplo, theta_zero * 1
    """

    def __init__(self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000):
        """
        Construtor do LogisticRegression

        :param l2_penalty: float
            The L2 regularization parameter
        :param alpha: float
            The learning rate
        :param max_iter: int
            The maximum number of iterations
        """
        # parâmetros
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter

        # atributos
        self.theta = None
        self.theta_zero = None
        self.cost_history = {}

    def fit(self, dataset: Dataset) -> 'LogisticRegression':
        """
        Faz o fit do modelo

        :param dataset: Dataset
            The dataset to fit the model to

        :return: self: LogisticRegression
            The fitted model
        """
        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n) # o tamanho é o número de features (temos um theta para cada uma das features)
        self.theta_zero = 0

        # threshold to stop gradient descent when cost function value stabilizes
        threshold = 0.0001

        # gradient descent
        for i in range(self.max_iter):
            # computing cost and updating cost_history
            self.cost_history[i] = self.cost(dataset=dataset)

            if i > 1 and (self.cost_history[i - 1] - self.cost_history[i] < threshold):
                break
            else:
                # predicted y
                y_pred = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)

                # computing and updating the gradient of the cost function with the learning rate
                gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.X)  # np.dot soma os valores das
                # colunas dos arrays de multiplicação
                # o learning rate é multiplicado por 1/m para normalizar o learning rate ao tamanho do dataset

                # computing the l2 penalty
                penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

                # updating the model parameters
                self.theta = self.theta - gradient - penalization_term
                self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

        return self

    def predict(self, dataset: Dataset) -> np.array:
        """
        Faz a previsão do output do dataset

        :param dataset: Dataset
            The dataset to predict the output of

        :return: predictions: np.array
            The predictions of the dataset
        """
        # apply, lista de compreensão ou usando uma mask
        y_pred = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)

        # # como fizemos na aula:
        # # convert the predictions to 0 or 1 (binarization)
        # mask = y_pred >= 0.5 # porque é o meio da função sigmoid (função que estima a probabilidade de um valor
        # # estar entre 0 e 1)
        # y_pred[mask] = 1
        # y_pred[~mask] = 0

        y_pred = [1 if x >= 0.5 else 0 for x in y_pred]

        return y_pred

    def score(self, dataset: Dataset) -> float:
        """
        Calcula o Mean Square Error do modelo no dataset

        :param dataset: Dataset
            The dataset to compute the MSE on

        :return: mse: float
            The Mean Square Error of the model
        """
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:
        """
        Calcula a cost function (J function) do modelo no dataset, usando a L2 regularization

        :param dataset: Dataset
            The dataset to compute the cost function on

        :return: cost: float
            The cost function of the model
        """
        # o predict do logístico faz o binário e é só para ajustar para classificação.
        # aqui queremos os valores estimados na regressão.
        predictions = sigmoid_function(np.dot(dataset.X, self.theta)+ self.theta_zero)
        cost = (-dataset.y * np.log(predictions)) - ((1 - dataset.y) * np.log(1 - predictions))
        cost = np.sum(cost) / dataset.shape()[0]
        cost = cost + (self.l2_penalty * np.sum(self.theta ** 2) / (2 * dataset.shape()[0])) # + porque, na fórmula,
        # multiplicamos por -1/m
        return cost

    def plot_cost_history(self):
        """
        Faz o plot do cost history em função do nº de iterações.
        """
        plt.plot(self.cost_history.keys(), self.cost_history.values())
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.show()

if __name__ == '__main__':
    # import dataset
    from si.data.dataset_module import Dataset
    from si.model_selection.split import train_test_split

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # fit the model
    model = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)
    model.fit(dataset_train)

    # compute the score
    score = model.score(dataset_test)
    print(f"Score: {score}")

    # compute the cost
    cost = model.cost(dataset_)
    print(f"Cost: {cost}")
