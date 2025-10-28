"""
NOMA Network Simulator

This module simulates Non-Orthogonal Multiple Access (NOMA) wireless networks.
It provides functionality to generate network scenarios, user distributions,
and channel conditions for testing resource allocation algorithms.
"""

import numpy as np
from typing import Tuple, List, Dict


class NOMASimulator:
    """Simulates NOMA wireless network scenarios."""
    
    def __init__(self, num_users: int = 4, num_channels: int = 2, num_subcarriers: int = 64):
        """
        Initialize NOMA Simulator.
        
        Args:
            num_users: Number of users in the network
            num_channels: Number of orthogonal channels
            num_subcarriers: Number of subcarriers per channel
        """
        self.num_users = num_users
        self.num_channels = num_channels
        self.num_subcarriers = num_subcarriers
        self.max_power = 1.0  # Normalized power
        
    def generate_channel_state_info(self) -> np.ndarray:
        """
        Generate random channel state information (CSI).
        
        Returns:
            Channel state matrix of shape (num_users, num_channels)
        """
        # Rayleigh fading channels
        real_part = np.random.randn(self.num_users, self.num_channels)
        imag_part = np.random.randn(self.num_users, self.num_channels)
        channels = np.abs(real_part + 1j * imag_part) / np.sqrt(2)
        return channels
    
    def generate_training_data(self, num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data for ML models.
        
        Args:
            num_samples: Number of training samples to generate
            
        Returns:
            X_train: Features of shape (num_samples, num_users * num_channels)
            y_train: Labels (optimal power allocation) of shape (num_samples, num_users)
        """
        X_train = []
        y_train = []
        
        for _ in range(num_samples):
            # Generate channel state info
            csi = self.generate_channel_state_info()
            X_train.append(csi.flatten())
            
            # Optimal power allocation (simplified: water-filling algorithm)
            power_alloc = self._waterfilling_algorithm(csi)
            y_train.append(power_alloc)
        
        return np.array(X_train), np.array(y_train)
    
    def _waterfilling_algorithm(self, channels: np.ndarray) -> np.ndarray:
        """
        Apply water-filling algorithm for optimal power allocation.
        
        Args:
            channels: Channel state matrix
            
        Returns:
            Optimal power allocation for each user
        """
        # Simplified water-filling: allocate power inversely proportional to channel gain
        avg_channels = np.mean(channels, axis=1)
        inv_channels = 1.0 / (avg_channels + 1e-8)
        power_alloc = (inv_channels / np.sum(inv_channels)) * self.max_power
        return power_alloc
    
    def calculate_sinr(self, channels: np.ndarray, power_alloc: np.ndarray,
                      noise_power: float = 0.1) -> np.ndarray:
        """
        Calculate Signal-to-Interference-plus-Noise Ratio (SINR).
        
        Args:
            channels: Channel state matrix
            power_alloc: Power allocation for each user
            noise_power: Noise power level
            
        Returns:
            SINR values for each user
        """
        signal_power = np.sum(channels * power_alloc[:, np.newaxis], axis=0)
        interference = np.sum(channels * power_alloc[:, np.newaxis], axis=0) - signal_power
        sinr = signal_power / (interference + noise_power)
        return sinr
    
    def calculate_sum_rate(self, sinr: np.ndarray) -> float:
        """
        Calculate total sum rate.
        
        Args:
            sinr: SINR values
            
        Returns:
            Sum rate in bits/second/Hz
        """
        bandwidth_per_channel = 1.0 / self.num_channels
        rate_per_channel = bandwidth_per_channel * np.log2(1 + sinr)
        sum_rate = np.sum(rate_per_channel)
        return sum_rate