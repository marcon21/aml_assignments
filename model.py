import numpy as np

activation_functions = {
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "relu": lambda x: np.maximum(0, x),
    "softmax": lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True),
}

activation_derivatives = {
    "sigmoid": lambda x: activation_functions["sigmoid"](x)
    * (1 - activation_functions["sigmoid"](x)),
    "relu": lambda x: np.where(x > 0, 1, 0),
    "softmax": lambda x: activation_functions["softmax"](x)
    * (1 - activation_functions["softmax"](x)),
}


class Layer:
    def __init__(self, input_s: int, output_s: int, activation: str):
        self.weights = np.random.randn(input_s, output_s) * np.sqrt(1.0 / input_s)
        self.biases = np.zeros((1, output_s))
        self.activation = activation

    def forward(self, x):
        x = x.reshape(1, self.weights.shape[0])
        self.input = x
        self.linear_output = np.dot(x, self.weights) + self.biases
        self.layer_output = activation_functions[self.activation](self.linear_output)
        return self.layer_output

    def backward(self, dA):
        activation_derivative = activation_derivatives[self.activation]
        dZ = dA * activation_derivative(self.layer_output)
        dW = np.dot(self.input.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        dA_prev = np.dot(dZ, self.weights.T)
        self.dW = dW
        self.db = db

        return dA_prev

    def update(self, learning_rate):
        self.weights -= learning_rate * self.dW
        self.biases -= learning_rate * self.db


class NN:
    def __init__(self, layers: list, lr: float = 0.01):
        self.layers = layers
        self.learning_rate = lr
        self.losses = []

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dA):
        for layer in reversed(self.layers):
            layer: Layer
            dA = layer.backward(dA)

    def update(
        self,
    ):
        for layer in self.layers:
            layer.update(self.learning_rate)

    def __call__(self, x):
        return self.forward(x)

    def train(self, X, epochs):
        for epoch in range(epochs):
            loss = []
            index = np.random.permutation(len(X))
            X = X[index]
            for x in X:
                y_hat = self.forward(x)
                error = y_hat - x
                loss.append(np.sum((error) ** 2))
                self.backward(error)
                self.update()
            loss = np.mean(loss)
            self.losses.append(loss)

            if epoch % (epochs / 10) == 0:
                print(f"Epoch {epoch} - Loss: {loss}")
