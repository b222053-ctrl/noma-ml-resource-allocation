"""
Main entry point for NOMA ML Resource Allocation application.

This script demonstrates how to use the NOMA simulator and ML-based resource
allocation algorithms for optimizing wireless network performance.
"""
import sys
import os
import numpy as np
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from src.noma_simulator import NOMASimulator
from src.resource_allocator import ResourceAllocator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def main():
    """Main function to run NOMA ML resource allocation."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='NOMA ML Resource Allocation - Machine Learning for Wireless Networks'
    )
    parser.add_argument('--num_users', type=int, default=4,
                        help='Number of users in the network (default: 4)')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='Number of channels (default: 2)')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of training samples (default: 1000)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        print("=" * 60)
        print("NOMA ML Resource Allocation System")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"  Number of Users: {args.num_users}")
        print(f"  Number of Channels: {args.num_channels}")
        print(f"  Training Samples: {args.num_samples}")
        print("=" * 60)
    
    try:
        # Initialize NOMA Simulator
        if args.verbose:
            print("\n[1/3] Initializing NOMA Simulator...")
        simulator = NOMASimulator(
            num_users=args.num_users,
            num_channels=args.num_channels
        )
        
        if args.verbose:
            print("      ✓ NOMA Simulator initialized successfully")
        
        # Generate training data
        if args.verbose:
            print(f"\n[2/3] Generating {args.num_samples} training samples...")
        X_train, y_train = simulator.generate_training_data(num_samples=args.num_samples)
        
        if args.verbose:
            print("      ✓ Training data generated successfully")
            print(f"        - Feature shape: {X_train.shape}")
            print(f"        - Label shape: {y_train.shape}")
        
        # Split the data into training and testing sets
        if args.verbose:
            print("\n[+] Splitting data into training and testing sets...")
        X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Normalize the training and testing data
        if args.verbose:
            print("\n[+] Normalizing training and testing data...")
        scaler = StandardScaler()
        X_train_split = scaler.fit_transform(X_train_split)
        X_test_split = scaler.transform(X_test_split)
        
        # Train a Random Forest model
        if args.verbose:
            print("\n[+] Training Random Forest model...")
        rf_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        rf_model.fit(X_train_split, y_train_split)
        rf_predictions = rf_model.predict(X_test_split)
        rf_mse = mean_squared_error(y_test_split, rf_predictions)

        # Evaluate training performance to check for overfitting
        rf_train_predictions = rf_model.predict(X_train_split)
        rf_train_mse = mean_squared_error(y_train_split, rf_train_predictions)

        if args.verbose:
            print(f"    ✓ Random Forest Training MSE: {rf_train_mse:.4f}")
            print(f"    ✓ Random Forest Testing MSE: {rf_mse:.4f}")
        
        # Train an SVM model using MultiOutputRegressor
        if args.verbose:
            print("\n[+] Training SVM model (multi-output)...")
        svm_model = MultiOutputRegressor(SVR(kernel='linear'))
        svm_model.fit(X_train_split, y_train_split)
        svm_predictions = svm_model.predict(X_test_split)
        svm_mse = mean_squared_error(y_test_split, svm_predictions)

        # Evaluate training performance to check for overfitting
        svm_train_predictions = svm_model.predict(X_train_split)
        svm_train_mse = mean_squared_error(y_train_split, svm_train_predictions)

        if args.verbose:
            print(f"    ✓ SVM Training MSE: {svm_train_mse:.4f}")
            print(f"    ✓ SVM Testing MSE: {svm_mse:.4f}")
        
        # Initialize Resource Allocator
        if args.verbose:
            print("\n[3/3] Initializing Resource Allocator...")
        allocator = ResourceAllocator()
        
        if args.verbose:
            print("      ✓ Resource Allocator initialized successfully")
        
        # Generate sample channel state information
        if args.verbose:
            print("\n[+] Generating sample channel state information...")
        csi = simulator.generate_channel_state_info()
        
        if args.verbose:
            print(f"    Channel State Information Shape: {csi.shape}")
            print(f"    Channel Matrix:\n{csi}")
        
        # Calculate optimal power allocation
        if args.verbose:
            print("\n[+] Calculating optimal power allocation (water-filling)...")
        optimal_power = simulator._waterfilling_algorithm(csi)
        
        if args.verbose:
            print(f"    Optimal Power Allocation: {optimal_power}")
            print(f"    Total Power Used: {sum(optimal_power):.4f}")
        
        # Calculate SINR
        if args.verbose:
            print("\n[+] Calculating SINR...")
        sinr = simulator.calculate_sinr(csi, optimal_power)
        
        # Calculate sum rate
        if args.verbose:
            print("\n[+] Calculating sum rate...")
        sum_rate = simulator.calculate_sum_rate(sinr)
        
        if args.verbose:
            print(f"    Sum Rate: {sum_rate:.4f} bits/s/Hz")

        # Compare ML-based allocation with traditional methods
        if args.verbose:
            print("\n[+] Comparing ML-based allocation with traditional methods...")

        # Transform CSI to match training feature structure
        if args.verbose:
            print("\n[+] Transforming and normalizing CSI...")
        csi_transformed = simulator.transform_csi_to_features(csi)
        csi_transformed = scaler.transform(csi_transformed)

        # Predict resource allocation using Random Forest
        rf_allocation = rf_model.predict(csi_transformed)
        rf_allocation = np.clip(rf_allocation, 0, 1)  # Clip to valid range
        rf_allocation /= np.sum(rf_allocation)  # Normalize to sum to 1
        rf_sinr = simulator.calculate_sinr(csi, rf_allocation)
        rf_sum_rate = simulator.calculate_sum_rate(rf_sinr)

        # Predict resource allocation using SVM
        svm_allocation = svm_model.predict(csi_transformed)
        svm_allocation = np.clip(svm_allocation, 0, 1)  # Clip to valid range
        svm_allocation /= np.sum(svm_allocation)  # Normalize to sum to 1
        svm_sinr = simulator.calculate_sinr(csi, svm_allocation)
        svm_sum_rate = simulator.calculate_sum_rate(svm_sinr)

        if args.verbose:
            print(f"    Traditional Method Sum Rate: {sum_rate:.4f} bits/s/Hz")
            print(f"    Random Forest Sum Rate: {rf_sum_rate:.4f} bits/s/Hz")
            print(f"    SVM Sum Rate: {svm_sum_rate:.4f} bits/s/Hz")
        
        # Evaluate performance metrics
        if args.verbose:
            print("\n[+] Evaluating performance metrics...")
        
        # Flatten the predictions to match the shape of optimal_power
        rf_allocation_flat = rf_allocation.flatten()
        svm_allocation_flat = svm_allocation.flatten()

        # Calculate MSE between ML-based and traditional allocations
        rf_mse_allocation = mean_squared_error(optimal_power, rf_allocation_flat)
        svm_mse_allocation = mean_squared_error(optimal_power, svm_allocation_flat)

        if args.verbose:
            print(f"    Random Forest Allocation MSE: {rf_mse_allocation:.4f}")
            print(f"    SVM Allocation MSE: {svm_mse_allocation:.4f}")
        
        # Visualize results
        if args.verbose:
            print("\n[+] Visualizing results...")
        
        methods = ['Traditional', 'Random Forest', 'SVM']
        sum_rates = [sum_rate, rf_sum_rate, svm_sum_rate]

        plt.bar(methods, sum_rates, color=['blue', 'green', 'orange'])
        plt.title('Sum Rate Comparison')
        plt.ylabel('Sum Rate (bits/s/Hz)')
        plt.show()
        
        print("\n" + "=" * 60)
        print("SIMULATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        if args.verbose:
            print("\nNext Steps:")
            print("1. Document results in a report.")
            print("2. Analyze the trade-offs between ML-based and traditional methods.")
            print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())