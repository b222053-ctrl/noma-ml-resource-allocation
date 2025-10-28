"""
Main entry point for NOMA ML Resource Allocation application.

This script demonstrates how to use the NOMA simulator and ML-based resource
allocation algorithms for optimizing wireless network performance.
"""

import sys
import argparse
from src.noma_simulator import NOMASimulator
from src.resource_allocator import ResourceAllocator


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
        
        if args.verbose:
            print(f"    SINR Values: {sinr}")
        
        # Calculate sum rate
        if args.verbose:
            print("\n[+] Calculating sum rate...")
        sum_rate = simulator.calculate_sum_rate(sinr)
        
        if args.verbose:
            print(f"    Sum Rate: {sum_rate:.4f} bits/s/Hz")
        
        print("\n" + "=" * 60)
        print("SIMULATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        if args.verbose:
            print("\nNext Steps:")
            print("1. Train ML models (SVM, Random Forest, Decision Tree, Gradient Descent)")
            print("2. Compare ML-based allocation with traditional methods")
            print("3. Evaluate performance metrics")
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