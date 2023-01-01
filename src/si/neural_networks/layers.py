import numpy as np

from si.statistics.sigmoid_function import sigmoid_function


class Dense:
    """
    Dense layer é uma camada onde cada neuron está ligado a todos os neurons da layer anterior.

    Parameters
    ----------
    input_size: int
        The number of inputs the layer will receive.
    output_size: int
        The number of outputs the layer will produce.

    Attributes
    ----------
    weights: np.ndarray
        The weights of the layer.
    bias: np.ndarray
        The bias of the layer.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Construtor da dense layer.

        :param input_size: int
            Número de inputs que a layer vai receber.
        :param output_size: int
            Número de outputs que a layer vai produzir.
        """
        # parameters
        self.input_size = input_size
        self.output_size = output_size

        # attributes
        self.X = None
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Método que calcula a forward pass da layer usando um dado input.

        :param X: np.ndarray
            Input da layer.

        :return: output: np.ndarray
            Output da layer com shape (1, output_size).
        """
        self.X = X
        return np.dot(X, self.weights) + self.bias

    def backward(self, error: np.ndarray, learning_rate: float = 0.01):
        """
        Método que calcula a backward pass da layer usando um dado input.

        :param error: np.ndarray
            Valores do erro para a loss function.
        :param learning_rate: float, optional
            Taxa de aprendizagem (default = 0.01).
        """
        error_to_propagate = np.dot(error, self.weights.T)

        # update weights and bias
        self.weights -= learning_rate * np.dot(self.X.T, error) # multiplicação de matrizes

        self.bias -= learning_rate * np.sum(error, axis=0) # bias: dimensões dos nodes; error: dimensões de samples +
        # nodes

        return error_to_propagate


class SigmoidActivation:
    """
    A sigmoid activation layer.
    """
    def __init__(self):
        """
        Construtor da sigmoid activation layer.

        :param input_size: int
            Número de inputs que a layer vai receber.
        :param output_size: int
            Número de outputs que a layer vai produzir.
        """
        self.X = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Método que calcula a forward pass da layer usando um dado input.

        :param X: np.ndarray
            Input da layer.

        :return: output: np.ndarray
            Output da layer com shape (1, output_size).
        """
        self.X = X
        return sigmoid_function(X)

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Método que calcula a backward pass da layer usando um dado input.

        :param error: np.ndarray
            Valores do erro para a loss function.
        :param learning_rate: float, optional
            Taxa de aprendizagem.
        """
        # multiplicação de cada elemento pela derivativa
        sigmoid_derivative = sigmoid_function(self.X) * (1 - sigmoid_function(self.X))

        error_to_propagate = error * sigmoid_derivative

        return error_to_propagate


class SoftMaxActivation:
    """
    A soft max activation layer.
    """
    def __init__(self):
        pass

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Método que calcula a probabilidade de cada classe.

        :param input_data: np.ndarray
            Input da layer.

        :return: np.ndarray
            Array com a probabilidade de cada classe.
        """
        zi_exp = np.exp(input_data - np.max(input_data))
        return zi_exp / (np.sum(zi_exp, axis=1, keepdims=True))


class ReLUActivation:
    """
    A rectified linear activation layer.
    """
    def __init__(self) -> None:
        self.input_data = None

    def forward(self, input_data: np.ndarray):
        """
        Método que calcula a relação linear retificada.

        :param input_data: np.ndarray
            Input data
        """
        self.input_data = input_data

        return np.maximum(0, input_data) # só considera a parte positiva do argumento

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Método que calcula a backward pass da layer usando um dado input.

        :param error: np.ndarray
            Valores do erro para a loss function.
        :param learning_rate: float, optional
            Taxa de aprendizagem.
        """
        relu_deriv = np.where(self.input_data > 0, 1, 0)

        error_to_propagate = error * relu_deriv

        return error_to_propagate


class LinearActivation:
    """
    A linear activation layer.
    """
    def __init__(self):
        pass

    def forward(input_data: np.ndarray) -> np.ndarray:
        """
        Método que calcula a ativação linear (no activation).

        :param input_data:
            Input data

        :return: Input data sem alterações.
        """
        return input_data
