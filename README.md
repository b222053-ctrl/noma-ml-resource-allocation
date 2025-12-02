# NOMA ML Resource Allocation

[![Python](https://img.shields.io/badge/Python-3.6%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ML](https://img.shields.io/badge/ML-Enabled-orange.svg)](https://scikit-learn.org/)

Machine learning-based resource allocation for NOMA (Non-Orthogonal Multiple Access) wireless networks.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [ML Models](#ml-models)
- [API Reference](#api-reference)
- [Performance Comparison](#performance-comparison)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements machine learning algorithms for optimizing resource allocation in NOMA wireless communication systems. NOMA is a promising technology for 5G and beyond, allowing multiple users to share the same frequency/time resources through power-domain multiplexing.

The system compares traditional optimization methods (like water-filling algorithm) with ML-based approaches to demonstrate the potential of machine learning in wireless network optimization.

## 🏗️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        NOMA ML Resource Allocation                          │
│                           System Architecture                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              Entry Points                                   │
├─────────────────────────────────┬───────────────────────────────────────────┤
│         src/main.py             │         test_all_models.py                │
│   (Main Application Runner)     │      (Model Testing & Validation)         │
└────────────────┬────────────────┴────────────────────┬──────────────────────┘
                 │                                     │
                 ▼                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Core Components                                  │
├─────────────────────────────────┬───────────────────────────────────────────┤
│      src/noma_simulator.py      │      src/resource_allocator.py            │
│   ┌─────────────────────────┐   │   ┌─────────────────────────────────┐     │
│   │ • NOMASimulator Class   │   │   │ • ResourceAllocator Class       │     │
│   │ • Channel State Info    │   │   │ • Allocation Strategies         │     │
│   │ • SINR Calculation      │   │   │ • Optimization Methods          │     │
│   │ • Sum Rate Calculation  │   │   └─────────────────────────────────┘     │
│   │ • Water-filling Algo    │   │                                           │
│   │ • Training Data Gen     │   │                                           │
│   └─────────────────────────┘   │                                           │
└─────────────────────────────────┴───────────────────────────────────────────┘
                 │
                 ▼
┌─��───────────────────────────────────────────────────────────────────────────┐
│                         ML Models Layer                                     │
│                       (src/ml_models/)                                      │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│  random_forest  │  svm_allocator  │ decision_tree   │ gradient_descent      │
│  _allocator.py  │     .py         │ _allocator.py   │ _allocator.py         │
│                 │                 │                 │                       │
│ RandomForest    │ SVM-based       │ Decision Tree   │ Gradient Descent      │
│ Regressor       │ Multi-Output    │ Classifier      │ Optimizer             │
└─────────────────┴─────────────────┴─────────────────┴───────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        External Dependencies                                │
├──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬───────────┤
│ sklearn  │  numpy   │  scipy   │tensorflow│  pandas  │matplotlib│  seaborn  │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴───────────┘
```

### Data Flow Architecture

```
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Channel    │───▶│  NOMASimulator   │───▶│  Training Data   │
│  Conditions  │    │  generate_data() │    │   (X, y)         │
└──────────────┘    └──────────────────┘    └────────┬─────────┘
                                                     │
                                                     ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Optimal    │◀───│   ML Models      │◀───│  Train/Test      │
│   Power      │    │   (RF, SVM, DT,  │    │  Split           │
│  Allocation  │    │    GD)           │    │                  │
└──────┬───────┘    └──────────────────┘    └──────────────────┘
       │
       ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│    SINR      │───▶│    Sum Rate      │───▶│   Performance    │
│ Calculation  │    │   Calculation    │    │   Comparison     │
└──────────────┘    └──────────────────┘    └───���──────────────┘
```

### Component Descriptions

| Component | File | Purpose |
|-----------|------|---------|
| **Main Application** | `src/main.py` | Entry point, orchestrates the entire workflow |
| **NOMA Simulator** | `src/noma_simulator.py` | Simulates NOMA wireless network behavior |
| **Resource Allocator** | `src/resource_allocator.py` | Handles resource allocation strategies |
| **Random Forest Model** | `src/ml_models/random_forest_allocator.py` | ML model using Random Forest |
| **SVM Model** | `src/ml_models/svm_allocator.py` | ML model using Support Vector Machine |
| **Decision Tree Model** | `src/ml_models/decision_tree_allocator.py` | ML model using Decision Tree |
| **Gradient Descent** | `src/ml_models/gradient_descent_allocator.py` | Optimization using Gradient Descent |
| **Model Tester** | `test_all_models.py` | Tests and compares all ML models |

## ✨ Features

- **NOMA Simulator**: Simulates Non-Orthogonal Multiple Access wireless network scenarios
- **Multiple ML Models**: Implements various machine learning algorithms for resource allocation
  - Random Forest Regressor
  - Support Vector Machine (SVM)
  - Decision Tree
  - Gradient Descent Optimizer
- **Water-filling Algorithm**: Traditional optimization method for comparison
- **Performance Metrics**: Calculates SINR, sum rate, and MSE for evaluation
- **Visualization**: Generates comparison charts using matplotlib

## 📁 Project Structure

```
noma-ml-resource-allocation/
├── src/
│   ├── __init__.py
│   ├── main.py                 # Main entry point
│   ├── noma_simulator.py       # NOMA network simulator
│   ├── resource_allocator.py   # Resource allocation logic
│   └── ml_models/
│       ├── __init__.py
│       ├── random_forest_allocator.py
│       ├── svm_allocator.py
│       ├── decision_tree_allocator.py
│       └── gradient_descent_allocator.py
├── test_all_models.py          # Model testing script
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── .gitignore
└── README.md
```

## 🚀 Installation

### Prerequisites

- Python 3.6 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/b222053-ctrl/noma-ml-resource-allocation.git
   cd noma-ml-resource-allocation
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package**
   ```bash
   pip install -e .
   ```

### Dependencies

| Package | Purpose |
|---------|---------|
| scikit-learn | Machine learning algorithms |
| numpy | Numerical computations |
| scipy | Scientific computing |
| matplotlib | Data visualization |
| tensorflow | Deep learning (optional) |
| pandas | Data manipulation |
| seaborn | Statistical visualization |
| jupyter | Interactive notebooks |
| scikit-optimize | Hyperparameter optimization |

## 💻 Usage

### Basic Usage

Run the main application with default settings:

```bash
python src/main.py
```

### With Verbose Output

```bash
python src/main.py --verbose
```

### Custom Configuration

```bash
python src/main.py --num_users 6 --num_channels 3 --num_samples 2000 --verbose
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_users` | int | 4 | Number of users in the network |
| `--num_channels` | int | 2 | Number of channels |
| `--num_samples` | int | 1000 | Number of training samples |
| `--verbose` | flag | False | Enable verbose output |

### Testing All Models

```bash
python test_all_models.py
```

## 🤖 ML Models

### 1. Random Forest Allocator
- **Algorithm**: Random Forest Regressor
- **Use Case**: Multi-output regression for power allocation
- **Strengths**: Handles non-linear relationships, robust to overfitting

### 2. SVM Allocator
- **Algorithm**: Support Vector Machine with MultiOutputRegressor
- **Use Case**: Linear and non-linear regression
- **Strengths**: Effective in high-dimensional spaces

### 3. Decision Tree Allocator
- **Algorithm**: Decision Tree Classifier/Regressor
- **Use Case**: Quick allocation decisions
- **Strengths**: Interpretable, fast inference

### 4. Gradient Descent Allocator
- **Algorithm**: Custom gradient descent optimization
- **Use Case**: Fine-tuned power allocation
- **Strengths**: Customizable learning rate and convergence

## 📚 API Reference

### NOMASimulator

```python
from src.noma_simulator import NOMASimulator

# Initialize simulator
simulator = NOMASimulator(num_users=4, num_channels=2)

# Generate training data
X_train, y_train = simulator.generate_training_data(num_samples=1000)

# Generate channel state information
csi = simulator.generate_channel_state_info()

# Calculate SINR
sinr = simulator.calculate_sinr(csi, power_allocation)

# Calculate sum rate
sum_rate = simulator.calculate_sum_rate(sinr)
```

### ResourceAllocator

```python
from src.resource_allocator import ResourceAllocator

# Initialize allocator
allocator = ResourceAllocator()

# Perform allocation (implementation specific)
```

## 📊 Performance Comparison

The system compares:

1. **Traditional Method**: Water-filling algorithm
2. **Random Forest**: ML-based allocation
3. **SVM**: Support Vector Machine allocation

Metrics evaluated:
- **Sum Rate** (bits/s/Hz)
- **Mean Squared Error** (MSE)
- **Training vs Testing Performance**

## 🔬 Technical Details

### NOMA Principles

NOMA allows multiple users to share the same resource block by:
1. Allocating different power levels to users
2. Using Successive Interference Cancellation (SIC) at receivers

### Key Equations

- **SINR**: Signal-to-Interference-plus-Noise Ratio
- **Sum Rate**: R = Σ log₂(1 + SINR)
- **Power Constraint**: Σ Pᵢ ≤ P_total

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

- **b222053-ctrl** - [GitHub Profile](https://github.com/b222053-ctrl)

## 🙏 Acknowledgments

- Inspired by research in NOMA wireless communication systems
- Built with scikit-learn and Python scientific computing libraries
