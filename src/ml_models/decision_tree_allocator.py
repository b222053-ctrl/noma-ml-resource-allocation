import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

class DecisionTreeAllocator:
    def __init__(self, max_depth=10):
        self.model = DecisionTreeRegressor(max_depth=max_depth)
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)

    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        return np.maximum(predictions, 0)

    def get_feature_importance(self):
        return self.model.feature_importances_

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        return {'mse': mse, 'mae': mae}