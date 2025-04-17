"""
Schema definitions for the V1 Spiderweb Bridge database.
"""

SCHEMA = {
    'nodes': '''
        CREATE TABLE IF NOT EXISTS nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            status TEXT NOT NULL,
            version TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            config TEXT,
            metadata TEXT
        )
    ''',
    'connections': '''
        CREATE TABLE IF NOT EXISTS connections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            connection_type TEXT NOT NULL,
            strength REAL DEFAULT 1.0,
            status TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT,
            FOREIGN KEY (source_id) REFERENCES nodes (node_id),
            FOREIGN KEY (target_id) REFERENCES nodes (node_id)
        )
    ''',
    'metrics': '''
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_type TEXT NOT NULL,
            value REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            node_id TEXT,
            connection_id INTEGER,
            metadata TEXT,
            FOREIGN KEY (node_id) REFERENCES nodes (node_id),
            FOREIGN KEY (connection_id) REFERENCES connections (id)
        )
    ''',
    'sync_events': '''
        CREATE TABLE IF NOT EXISTS sync_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            status TEXT NOT NULL,
            source_version TEXT NOT NULL,
            target_version TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            details TEXT,
            error_message TEXT
        )
    '''
} 