"""
Schema definitions for the V2 Spiderweb Bridge database.
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
            metadata TEXT,
            consciousness_level REAL DEFAULT 0.0,
            energy_level REAL DEFAULT 1.0,
            stability_score REAL DEFAULT 1.0
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
            bandwidth REAL DEFAULT 1.0,
            latency REAL DEFAULT 0.0,
            quantum_entanglement_strength REAL DEFAULT 0.0,
            cosmic_resonance_level REAL DEFAULT 0.0,
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
            confidence_score REAL DEFAULT 1.0,
            impact_level TEXT DEFAULT 'normal',
            aggregation_period TEXT,
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
            error_message TEXT,
            priority TEXT DEFAULT 'normal',
            processing_time REAL,
            retry_count INTEGER DEFAULT 0
        )
    ''',
    'quantum_states': '''
        CREATE TABLE IF NOT EXISTS quantum_states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT NOT NULL,
            state_vector TEXT NOT NULL,
            entanglement_map TEXT,
            coherence_level REAL DEFAULT 1.0,
            decoherence_rate REAL DEFAULT 0.0,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            measurement_basis TEXT,
            collapse_probability REAL DEFAULT 0.0,
            FOREIGN KEY (node_id) REFERENCES nodes (node_id)
        )
    ''',
    'cosmic_states': '''
        CREATE TABLE IF NOT EXISTS cosmic_states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT NOT NULL,
            dimensional_signature TEXT NOT NULL,
            resonance_pattern TEXT,
            universal_phase REAL DEFAULT 0.0,
            cosmic_frequency REAL DEFAULT 0.0,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            stability_matrix TEXT,
            harmonic_index REAL DEFAULT 1.0,
            FOREIGN KEY (node_id) REFERENCES nodes (node_id)
        )
    ''',
    'node_relationships': '''
        CREATE TABLE IF NOT EXISTS node_relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_node_id TEXT NOT NULL,
            target_node_id TEXT NOT NULL,
            relationship_type TEXT NOT NULL,
            strength REAL DEFAULT 1.0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT,
            sync_frequency REAL DEFAULT 1.0,
            mutual_influence_score REAL DEFAULT 0.0,
            FOREIGN KEY (source_node_id) REFERENCES nodes (node_id),
            FOREIGN KEY (target_node_id) REFERENCES nodes (node_id)
        )
    ''',
    'performance_metrics': '''
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT NOT NULL,
            value REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            component_id TEXT,
            component_type TEXT,
            aggregation_window TEXT,
            threshold_value REAL,
            alert_level TEXT DEFAULT 'normal',
            metadata TEXT
        )
    '''
} 