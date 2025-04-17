# Neural Network Implementation - Version 1

This directory contains the first version of our neural network implementation, featuring a basic feedforward neural network with backpropagation and a distributed node system.

## Structure

```
v1/
├── __init__.py           # Package initialization
├── core/                 # Core neural network implementation
│   ├── __init__.py
│   └── neural_network.py # Main neural network class
├── utils/                # Utility classes
│   ├── __init__.py
│   └── data_processing.py # Data processing and model evaluation
├── nodes/                # Node system implementation
│   ├── __init__.py
│   ├── node_implementation.py # HybridNode and CentralNode classes
│   └── example.py        # Node system example
├── example.py            # Basic neural network example
└── README.md             # This file
```

## Features

- Feedforward neural network with customizable architecture
- Backpropagation with mini-batch gradient descent
- ReLU activation function
- Mean squared error loss
- Data preprocessing with standardization
- Model evaluation with various metrics
- Cross-validation support
- Model weight saving and loading
- Distributed node system with:
  - Hybrid nodes for local processing
  - Central node for coordination
  - Weight averaging for model aggregation
  - Ring topology for node connections

## Dependencies

- Python 3.6+
- NumPy
- scikit-learn

## Usage

1. Install dependencies:
```bash
pip install numpy scikit-learn
```

2. Run the basic example:
```bash
python example.py
```

3. Run the node system example:
```bash
python nodes/example.py
```

## Core Components

### NeuralNetwork Class

The main neural network implementation with the following key methods:
- `__init__`: Initialize the network with specified layer sizes
- `forward`: Perform forward propagation
- `backward`: Perform backpropagation
- `train`: Train the network on a batch of data
- `predict`: Make predictions
- `save_weights`: Save model weights
- `load_weights`: Load model weights

### DataProcessor Class

Utility class for data preprocessing:
- `preprocess_data`: Scale and split data into train/test sets
- `transform`: Transform new data using fitted scaler
- `inverse_transform`: Transform scaled data back to original scale

### ModelEvaluator Class

Utility class for model evaluation:
- `evaluate`: Calculate various metrics on test data
- `cross_validate`: Perform k-fold cross-validation

### HybridNode Class

Distributed node implementation with:
- Local neural network instance
- Data storage and processing
- Node-to-node connections
- Local training capabilities
- Weight sharing interface

### CentralNode Class

Central coordination node with:
- Hybrid node registration
- Training coordination
- Weight aggregation
- Status monitoring
- Distributed learning management

## Example

### Basic Neural Network
```python
from core.neural_network import NeuralNetwork
from utils.data_processing import DataProcessor, ModelEvaluator

# Initialize network
layer_sizes = [10, 64, 32, 1]  # Input -> Hidden -> Hidden -> Output
model = NeuralNetwork(layer_sizes, learning_rate=0.01)

# Preprocess data
processor = DataProcessor()
(X_train, y_train), (X_test, y_test) = processor.preprocess_data(X, y)

# Train model
for epoch in range(100):
    model.train(X_train.T, y_train.T)

# Evaluate
evaluator = ModelEvaluator(model)
metrics = evaluator.evaluate(X_test.T, y_test.T)
```

### Distributed Node System
```python
from nodes.node_implementation import HybridNode, CentralNode

# Initialize central node
central_node = CentralNode()

# Create and register hybrid nodes
node1 = HybridNode("node1", [10, 64, 32, 1])
node2 = HybridNode("node2", [10, 64, 32, 1])
central_node.register_hybrid_node(node1)
central_node.register_hybrid_node(node2)

# Connect nodes
node1.connect_to_node("node2")
node2.connect_to_node("node1")

# Store local data
node1.store_local_data("data1", X1, y1)
node2.store_local_data("data2", X2, y2)

# Perform coordinated training
metrics = central_node.coordinate_training(n_rounds=10)
```

## Notes

- This is a basic implementation suitable for binary classification tasks
- The network uses ReLU activation and mean squared error loss
- Data should be properly scaled before training
- The implementation supports mini-batch training
- Model weights can be saved and loaded for later use
- The node system implements a basic form of federated learning
- Nodes can be connected in various topologies (currently implements ring topology)
- The central node coordinates training through weight averaging 