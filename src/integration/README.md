# Backend Integration System

This system provides the central integration point for all backend components of the Lumina Neural Network System. It coordinates the initialization, connection, and communication between all major subsystems.

## Features

- Centralized component management
- Version compatibility handling
- Asynchronous event processing
- Comprehensive metrics collection
- Graceful shutdown handling
- Configuration management
- Mock mode for testing
- **State persistence and recovery**
- **Automatic state saving**
- **Version control for system states**
- **Distributed learning and knowledge sharing**
- **Inter-node teaching and collaboration**
- **Automated knowledge transfer**

## Components

The system integrates the following major components:

1. **Version Bridge**
   - Manages version compatibility
   - Handles data transformation
   - Provides version-specific APIs

2. **Language Memory**
   - Neural linguistic processing
   - Memory synthesis
   - Pattern recognition

3. **Neural Playground**
   - Neural network simulation
   - Learning algorithms
   - Pattern generation

4. **Visualization**
   - Real-time visualization
   - Pattern display
   - Metrics visualization

5. **Consciousness**
   - Quantum consciousness
   - Cosmic consciousness
   - Temporal consciousness

6. **Database**
   - State persistence
   - Metrics storage
   - Configuration management

7. **Persistence System**
   - Manages system-wide state persistence
   - Automatic state saving
   - Version control for system states
   - Component-specific state storage
   - State integrity verification
   - Automatic pruning of old versions
   - SQLite database for state metadata
   - JSON storage for component states
   - Checksum verification
   - Transaction support

8. **Distributed Learning System**
   - Node-to-node teaching
   - Knowledge distillation
   - Collaborative learning
   - Dynamic teacher-student relationships
   - Automated knowledge transfer
   - Real-time model adaptation
   - Peer-to-peer learning
   - Ensemble knowledge sharing
   - Performance monitoring
   - Learning optimization

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure the system (see Configuration section)

## Usage

### Basic Usage

```python
from src.integration.main import BackendIntegrationSystem

# Create and run the system
system = BackendIntegrationSystem()
await system.initialize()
await system.run()
```

### Distributed Learning Setup

```python
from src.integration.ml.distributed_learning import (
    DistributedLearningSystem,
    LearningNode,
    NodeConfig
)

# Create learning nodes
node1 = LearningNode(NodeConfig(node_id="node1"))
node2 = LearningNode(NodeConfig(node_id="node2"))
node3 = LearningNode(NodeConfig(node_id="node3"))

# Add nodes to the system
learning_system = DistributedLearningSystem()
learning_system.add_node(node1)
learning_system.add_node(node2)
learning_system.add_node(node3)

# Start distributed learning
await learning_system.start_learning(training_data)
```

### Knowledge Transfer

```python
# Get node states
node_states = learning_system.get_system_state()

# Check learning progress
for node_id, state in node_states['nodes'].items():
    print(f"Node {node_id}:")
    print(f"  Knowledge level: {state['knowledge_level']:.2f}")
    print(f"  Students: {state['num_students']}")
    print(f"  Teachers: {state['num_teachers']}")
```

## Configuration

The system can be configured through the `config.py` file or by passing a custom configuration dictionary. Key configuration sections include:

1. **Version Bridge**
   - Version range
   - Compatibility rules
   - Mock mode settings

2. **Language Memory**
   - LLM weight
   - Memory size
   - Cache settings

3. **Neural Playground**
   - Network size
   - Learning parameters
   - Activation thresholds

4. **Visualization**
   - Update frequency
   - Quality settings
   - History size

5. **Consciousness**
   - Quantum settings
   - Cosmic settings
   - Temporal settings

6. **Database**
   - Type and path
   - Cache size
   - Backup interval

7. **System**
   - Logging level
   - Thread count
   - Event queue size

### Persistence Configuration

```yaml
persistence:
  auto_save_interval: 300  # seconds
  max_versions: 100
  storage_path: "storage"
  db_path: "storage/persistence.db"
  collector_config:
    include_metrics: true
    include_history: true
    max_history_items: 1000
```

### Component Configuration

```yaml
components:
  bridge:
    version_compatibility: true
    data_transformation: true
    
  neural_seed:
    growth_rate: 0.5
    stability_threshold: 0.8
    
  autowiki:
    article_generation: true
    learning_enabled: true
    
  spiderweb:
    quantum_field_strength: 0.8
    cosmic_field_strength: 0.7
```

### Distributed Learning Configuration

```yaml
distributed_learning:
  node_config:
    learning_rate: 0.001
    knowledge_threshold: 0.8
    teach_interval: 100
    max_students: 5
    distillation_temperature: 2.0
    collaboration_weight: 0.3
```

### Model Configuration

```yaml
model:
  type: "transformer"
  hidden_size: 256
  num_layers: 4
  num_heads: 8
  dropout: 0.1
```

### Learning System Configuration

```yaml
learning_system:
  auto_scale: true
  min_nodes: 3
  max_nodes: 10
  learning_rate: 0.001
  batch_size: 32
  evaluation_interval: 100
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

The project uses:
- Black for code formatting
- isort for import sorting
- mypy for type checking
- flake8 for linting

Run all checks:
```bash
black .
isort .
mypy .
flake8
```

## Architecture

### Distributed Learning Architecture

The distributed learning system implements a dynamic teacher-student network where:

1. **Knowledge Transfer**
   - Nodes can act as both teachers and students
   - Knowledge is transferred through distillation
   - Teaching relationships adapt based on performance

2. **Collaborative Learning**
   - Nodes share knowledge through ensemble learning
   - Peer-to-peer knowledge exchange
   - Dynamic collaboration weights

3. **Performance Monitoring**
   - Real-time accuracy tracking
   - Learning rate adaptation
   - Knowledge level assessment

4. **Optimization**
   - Automatic teacher selection
   - Dynamic batch sizing
   - Resource-aware scaling

## Distributed Learning System

The distributed learning system enables autonomous knowledge sharing and learning between nodes through a sophisticated teacher-student network architecture.

### Core Components

1. **Learning Nodes**
   ```python
   from src.integration.ml.distributed_learning import LearningNode, NodeConfig
   
   # Create a learning node
   node = LearningNode(
       NodeConfig(
           node_id="node1",
           model_type="transformer",
           learning_rate=0.001,
           knowledge_threshold=0.8,
           max_students=5
       )
   )
   ```

2. **Knowledge Transfer**
   ```python
   # Teaching interaction
   await teacher_node.teach(student_node, data)
   
   # Learning from teacher
   metrics = await student_node.learn(data, teacher=teacher_node)
   print(f"Learning metrics: {metrics}")
   ```

3. **Collaborative Learning**
   ```python
   # Peer collaboration
   peers = [node2, node3, node4]
   metrics = await node1.collaborate(peers, data)
   print(f"Collaboration metrics: {metrics}")
   ```

### Features

1. **Knowledge Distillation**
   - Temperature-scaled softmax
   - KL divergence loss
   - Gradual knowledge transfer
   ```python
   # Configure distillation
   config = NodeConfig(
       node_id="student1",
       distillation_temperature=2.0,
       collaboration_weight=0.3
   )
   ```

2. **Dynamic Teacher-Student Relationships**
   - Automatic teacher selection
   - Performance-based pairing
   - Maximum student limits
   ```python
   # System automatically manages relationships
   if len(node.students) < node.config.max_students:
       await node.teach(potential_student, data)
   ```

3. **Collaborative Ensemble Learning**
   - Peer knowledge aggregation
   - Weighted ensemble predictions
   - Cross-node validation
   ```python
   # Ensemble learning
   peer_predictions = []
   for peer in peers:
       predictions = await peer.predict(data)
       peer_predictions.append(predictions)
   ensemble = average_predictions(peer_predictions)
   ```

4. **Performance Monitoring**
   - Real-time metrics tracking
   - Learning history
   - Teaching effectiveness
   ```python
   # Get node performance
   state = node.get_knowledge_state()
   print(f"Knowledge level: {state['knowledge_level']}")
   print(f"Recent history: {state['learning_history']}")
   ```

### Configuration

```yaml
distributed_learning:
  node:
    learning_rate: 0.001
    knowledge_threshold: 0.8
    teach_interval: 100
    max_students: 5
    distillation_temperature: 2.0
    collaboration_weight: 0.3
  
  system:
    min_nodes: 3
    max_nodes: 10
    evaluation_interval: 100
    auto_scale: true
```

### System Management

1. **Node Management**
   ```python
   # Add/remove nodes
   system.add_node(new_node)
   system.remove_node("node1")
   
   # Get system state
   state = system.get_system_state()
   print(f"Total nodes: {state['num_nodes']}")
   print(f"Learning events: {state['total_learning_events']}")
   ```

2. **Learning Control**
   ```python
   # Start distributed learning
   await system.start_learning(training_data)
   
   # Monitor progress
   for node_id, node_state in system.get_system_state()['nodes'].items():
       print(f"Node {node_id}: {node_state['knowledge_level']:.2f}")
   ```

### Error Handling

The system includes robust error handling:
```python
try:
    await node.learn(data)
except Exception as e:
    logger.error(f"Learning error in node {node.config.node_id}: {e}")
    await asyncio.sleep(5)  # Retry delay
```

### Performance Optimization

1. **Resource Management**
   - Asynchronous operations
   - Batch processing
   - Memory-efficient knowledge transfer

2. **Learning Optimization**
   - Dynamic learning rates
   - Adaptive batch sizes
   - Performance-based routing

3. **System Scaling**
   - Automatic node management
   - Load balancing
   - Resource allocation

### Monitoring and Metrics

1. **Node Metrics**
   - Learning accuracy
   - Teaching effectiveness
   - Knowledge level
   - Peer collaboration success

2. **System Metrics**
   - Total learning events
   - Teaching events
   - Node distribution
   - Overall performance

### Best Practices

1. **Node Configuration**
   - Start with small teacher-student ratios
   - Use appropriate distillation temperatures
   - Balance collaboration weights

2. **System Setup**
   - Initialize with sufficient nodes
   - Enable auto-scaling
   - Monitor resource usage

3. **Performance Tuning**
   - Adjust learning rates based on metrics
   - Optimize batch sizes
   - Fine-tune collaboration weights

### Example Workflow

```python
from src.integration.ml.distributed_learning import (
    DistributedLearningSystem,
    LearningNode,
    NodeConfig
)

# Create system
system = DistributedLearningSystem()

# Initialize nodes
nodes = [
    LearningNode(NodeConfig(
        node_id=f"node{i}",
        learning_rate=0.001,
        max_students=5
    ))
    for i in range(5)
]

# Add nodes to system
for node in nodes:
    system.add_node(node)

# Start learning
async def main():
    await system.start_learning(training_data)
    
    # Monitor progress
    while True:
        state = system.get_system_state()
        print(f"Total learning events: {state['total_learning_events']}")
        await asyncio.sleep(60)

# Run system
asyncio.run(main())
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 