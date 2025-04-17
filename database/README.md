# Neural Network State Database

This module provides database functionality for managing neural network states in the Lumina system.

## Overview

The database module uses SQLite to store and manage neural network states, including:
- State metrics
- Version data
- Timestamps
- State types

## Usage

```python
from database import DatabaseManager

# Get database instance
db = DatabaseManager()

# Save a neural state
db.save_neural_state("training", metrics_dict, version_data)

# Retrieve a state
state = db.get_neural_state(state_id)

# Get all states
all_states = db.get_all_states()

# Delete a state
db.delete_neural_state(state_id)
```

## Database Schema

The `neural_states` table has the following schema:
- `id` (INTEGER): Primary key
- `timestamp` (TIMESTAMP): When the state was saved
- `state_type` (TEXT): Type of neural state
- `metrics` (TEXT): JSON serialized metrics data
- `version_data` (TEXT): JSON serialized version data

## Error Handling

All database operations include error handling and logging. Errors are logged to help with debugging and maintaining data integrity. 