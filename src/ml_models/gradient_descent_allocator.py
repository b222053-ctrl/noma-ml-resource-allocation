import numpy as np

class GradientDescentAllocator:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.loss_history = []

    def train(self, X_train, y_train):
        m, n = X_train.shape
        self.weights = np.zeros(n)
        self.bias = 0
        self.loss_history = []
        for iteration in range(self.max_iterations):
            y_pred = X_train.dot(self.weights) + self.bias
            loss = (1/(2*m)) * np.sum((y_pred - y_train)**2)
            self.loss_history.append(loss)
            dw = (1/m) * X_train.T.dot(y_pred - y_train)
            db = (1/m) * np.sum(y_pred - y_train)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            if iteration > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.tolerance:
                break

    def predict(self, X_test):
        predictions = X_test.dot(self.weights) + self.bias
        predictions = np.maximum(predictions, 0)
        return predictions

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        return {'mse': mse, 'mae': mae}

    def get_loss_history(self):
        return self.loss_history