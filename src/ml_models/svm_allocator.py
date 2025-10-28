import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

class SVMAllocator:
    def __init__(self):
        self.model = SVR()
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)

    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        mse = np.mean((predictions - y_test) ** 2)
        return mse
