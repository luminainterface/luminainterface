SCHEMA = {
    'neural_states': '''
        CREATE TABLE IF NOT EXISTS neural_states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            state_data TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''',
    'metrics': '''
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_type TEXT NOT NULL,
            metric_value REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    '''
} 