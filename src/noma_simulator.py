import numpy as np

class NOMASimulator:
    def __init__(self, num_users, num_channels):
        """
        Initialize the NOMA Simulator.

        Args:
            num_users (int): Number of users in the network.
            num_channels (int): Number of channels in the network.
        """
        self.num_users = num_users
        self.num_channels = num_channels

    def generate_training_data(self, num_samples):
        """
        Generate training data for the NOMA system.

        Args:
            num_samples (int): Number of training samples to generate.

        Returns:
            tuple: A tuple (X_train, y_train) where:
                - X_train is the feature matrix of shape (num_samples, num_features).
                - y_train is the label matrix of shape (num_samples, num_users).
        """
        # Example: Randomly generate features and labels
        X_train = np.random.rand(num_samples, self.num_users * self.num_channels)
        y_train = np.random.rand(num_samples, self.num_users)
        return X_train, y_train

    def generate_channel_state_info(self):
        """
        Generate a sample Channel State Information (CSI) matrix.

        Returns:
            numpy.ndarray: A CSI matrix of shape (num_users, num_channels).
        """
        return np.random.rand(self.num_users, self.num_channels)

    def _waterfilling_algorithm(self, csi):
        """
        Perform the water-filling algorithm for power allocation.

        Args:
            csi (numpy.ndarray): The Channel State Information matrix.

        Returns:
            numpy.ndarray: Optimal power allocation for each user.
        """
        num_users, num_channels = csi.shape
        power_allocation = np.random.rand(num_users)  # Example: Random allocation
        power_allocation /= np.sum(power_allocation)  # Normalize to sum to 1
        return power_allocation

    def calculate_sinr(self, csi, power_allocation):
        """
        Calculate the Signal-to-Interference-plus-Noise Ratio (SINR).

        Args:
            csi (numpy.ndarray): The Channel State Information matrix.
            power_allocation (numpy.ndarray): Power allocation for each user.

        Returns:
            numpy.ndarray: SINR values for each user.
        """
        return np.random.rand(self.num_users) * 10  # Example: Random SINR values

    def calculate_sum_rate(self, sinr):
        """
        Calculate the sum rate of the system.

        Args:
            sinr (numpy.ndarray): SINR values for each user.

        Returns:
            float: The sum rate of the system.
        """
        return np.sum(np.log2(1 + sinr))  # Example: Sum rate calculation

    def transform_csi_to_features(self, csi):
        """
        Transform the CSI matrix into the same feature structure as the training data.

        Args:
            csi (numpy.ndarray): The Channel State Information matrix (num_users x num_channels).

        Returns:
            numpy.ndarray: Transformed feature matrix matching the training data structure.
        """
        # Flatten the CSI matrix to create a feature vector
        features = csi.flatten()  # Flatten the CSI matrix
        return features.reshape(1, -1)  # Reshape to (1, num_features)