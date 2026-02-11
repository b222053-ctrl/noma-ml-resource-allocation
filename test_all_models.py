import time
import numpy as np
from src.ml_models import SVMAllocator, RandomForestAllocator, DecisionTreeAllocator, GradientDescentAllocator
from src.noma_simulator import NOMASimulator

# SVM Model Testing
def run_svm(X_train, X_test, y_train, y_test):
    print("\n" + "="*50)
    print("Testing SVM Allocator")
    print("="*50)
    start_time = time.time()
    
    model = SVMAllocator()
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    
    print(f"SVM Model MSE: {metrics['mse']:.6f}")
    print(f"SVM Model MAE: {metrics['mae']:.6f}")
    print(f"SVM Execution Time: {time.time() - start_time:.4f} seconds")
    
    return metrics

# Random Forest Model Testing
def run_random_forest(X_train, X_test, y_train, y_test):
    print("\n" + "="*50)
    print("Testing Random Forest Allocator")
    print("="*50)
    start_time = time.time()
    
    model = RandomForestAllocator(n_estimators=100)
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    
    print(f"Random Forest Model MSE: {metrics['mse']:.6f}")
    print(f"Random Forest Model MAE: {metrics['mae']:.6f}")
    print(f"Random Forest Execution Time: {time.time() - start_time:.4f} seconds")
    
    return metrics

# Decision Tree Model Testing
def run_decision_tree(X_train, X_test, y_train, y_test):
    print("\n" + "="*50)
    print("Testing Decision Tree Allocator")
    print("="*50)
    start_time = time.time()
    
    model = DecisionTreeAllocator(max_depth=10)
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    
    print(f"Decision Tree Model MSE: {metrics['mse']:.6f}")
    print(f"Decision Tree Model MAE: {metrics['mae']:.6f}")
    print(f"Decision Tree Execution Time: {time.time() - start_time:.4f} seconds")
    
    return metrics

# Gradient Descent Model Testing
def run_gradient_descent(X_train, X_test, y_train, y_test):
    print("\n" + "="*50)
    print("Testing Gradient Descent Allocator")
    print("="*50)
    start_time = time.time()
    
    model = GradientDescentAllocator(learning_rate=0.01, max_iterations=1000)
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    
    print(f"Gradient Descent Model MSE: {metrics['mse']:.6f}")
    print(f"Gradient Descent Model MAE: {metrics['mae']:.6f}")
    print(f"Gradient Descent Execution Time: {time.time() - start_time:.4f} seconds")
    print(f"Number of iterations: {len(model.get_loss_history())}")
    
    return metrics

# Main testing function
def main():
    print("="*50)
    print("NOMA Resource Allocation - ML Models Testing")
    print("="*50)
    
    # Generate realistic NOMA training data
    print("\nGenerating NOMA training data...")
    simulator = NOMASimulator(num_users=4, num_channels=2)
    X_train, y_train = simulator.generate_training_data(num_samples=1000)
    X_test, y_test = simulator.generate_training_data(num_samples=200)
    
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")

    # Testing all models
    svm_metrics = run_svm(X_train, X_test, y_train, y_test)
    rf_metrics = run_random_forest(X_train, X_test, y_train, y_test)
    dt_metrics = run_decision_tree(X_train, X_test, y_train, y_test)
    gd_metrics = run_gradient_descent(X_train, X_test, y_train, y_test)

    # Summary of Results
    print("\n" + "="*50)
    print("Performance Comparison Summary")
    print("="*50)
    print(f"{'Model':<25} {'MSE':<15} {'MAE':<15}")
    print("-"*50)
    print(f"{'SVM':<25} {svm_metrics['mse']:<15.6f} {svm_metrics['mae']:<15.6f}")
    print(f"{'Random Forest':<25} {rf_metrics['mse']:<15.6f} {rf_metrics['mae']:<15.6f}")
    print(f"{'Decision Tree':<25} {dt_metrics['mse']:<15.6f} {dt_metrics['mae']:<15.6f}")
    print(f"{'Gradient Descent':<25} {gd_metrics['mse']:<15.6f} {gd_metrics['mae']:<15.6f}")
    print("="*50)
    
    # Determine best model
    models = {
        'SVM': svm_metrics['mse'],
        'Random Forest': rf_metrics['mse'],
        'Decision Tree': dt_metrics['mse'],
        'Gradient Descent': gd_metrics['mse']
    }
    best_model = min(models, key=models.get)
    print(f"\nBest performing model (lowest MSE): {best_model}")
    print("="*50)

if __name__ == "__main__":
    main()