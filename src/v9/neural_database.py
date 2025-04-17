#!/usr/bin/env python3
"""
Neural Network Database (v9)

This module provides database functionality for storing, retrieving, and analyzing
data from the Neural-Breathing integration system. It tracks neural growth,
consciousness metrics, and breathing patterns over time to enable insights
into the relationships between breathing and neural development.

Key features:
- Session data storage and retrieval
- Neural growth tracking across breathing patterns
- Consciousness metric analysis
- Trend identification and reporting
- Data visualization capabilities
"""

import os
import json
import sqlite3
import time
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("v9.neural_database")

class NeuralDatabase:
    """
    Database for neural network data storage and analysis
    
    This class manages persistent storage of neural network state,
    breathing patterns, and consciousness metrics. It allows for
    historical analysis and visualization of neural development.
    """
    
    def __init__(self, db_path: str = "neural_database.db"):
        """
        Initialize the neural database
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._initialize_database()
        logger.info(f"Neural database initialized at {db_path}")
    
    def _initialize_database(self):
        """Create necessary database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create sessions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            duration INTEGER,
            play_type TEXT,
            intensity REAL,
            breathing_pattern TEXT,
            neurons_initial INTEGER,
            neurons_final INTEGER,
            consciousness_peak REAL,
            patterns_detected INTEGER,
            total_activations INTEGER,
            metadata TEXT
        )
        ''')
        
        # Create neural_metrics table for time-series data
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS neural_metrics (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            timestamp TEXT,
            metric_type TEXT,
            metric_value REAL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
        ''')
        
        # Create growth_events table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS growth_events (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            timestamp TEXT,
            growth_state TEXT,
            neurons_created INTEGER,
            neurons_pruned INTEGER,
            regions_formed INTEGER,
            breath_pattern TEXT,
            breath_coherence REAL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
        ''')
        
        # Create consciousness_markers table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS consciousness_markers (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            timestamp TEXT,
            consciousness_level REAL,
            pattern_type TEXT,
            breath_state TEXT,
            breath_pattern TEXT,
            description TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_session(self, session_data: Dict) -> str:
        """
        Store a neural playground session in the database
        
        Args:
            session_data: Dictionary containing session data
            
        Returns:
            Session ID
        """
        # Generate a session ID if not provided
        session_id = session_data.get("session_id", f"session_{int(time.time())}")
        
        # Extract breathing data
        breathing_data = session_data.get("breathing_data", {})
        breathing_pattern = breathing_data.get("pattern", "unknown")
        
        # Extract main metrics
        timestamp = datetime.datetime.now().isoformat()
        duration = session_data.get("duration", 0)
        play_type = session_data.get("play_type", "unknown")
        intensity = session_data.get("intensity", 0.0)
        consciousness_peak = session_data.get("consciousness_peak", 0.0)
        patterns_detected = session_data.get("patterns_detected", 0)
        total_activations = session_data.get("total_activations", 0)
        
        # Extract network size data
        network_size = session_data.get("network_size", {})
        neurons_initial = network_size.get("initial", 0)
        neurons_final = network_size.get("final", 0)
        
        # All other data as JSON
        metadata = json.dumps({
            k: v for k, v in session_data.items() 
            if k not in ["session_id", "timestamp", "duration", "play_type", 
                         "intensity", "breathing_data", "consciousness_peak", 
                         "patterns_detected", "total_activations", "network_size"]
        })
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT OR REPLACE INTO sessions 
        (id, timestamp, duration, play_type, intensity, breathing_pattern, 
         neurons_initial, neurons_final, consciousness_peak, patterns_detected, 
         total_activations, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id, timestamp, duration, play_type, intensity, breathing_pattern,
            neurons_initial, neurons_final, consciousness_peak, patterns_detected,
            total_activations, metadata
        ))
        
        # Store growth events if available
        if "brain_growth" in session_data:
            growth_data = session_data["brain_growth"]
            growth_id = f"growth_{int(time.time())}"
            
            cursor.execute('''
            INSERT INTO growth_events 
            (id, session_id, timestamp, growth_state, neurons_created, neurons_pruned, 
             regions_formed, breath_pattern, breath_coherence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                growth_id, session_id, timestamp, growth_data.get("growth_state", "unknown"),
                growth_data.get("neurons_created", 0), growth_data.get("neurons_pruned", 0),
                growth_data.get("regions_formed", 0), breathing_pattern,
                breathing_data.get("coherence", 0.0)
            ))
        
        # Store consciousness history if available
        if "consciousness_history" in session_data:
            history = session_data["consciousness_history"]
            for i, level in enumerate(history):
                marker_id = f"cm_{session_id}_{i}"
                marker_time = datetime.datetime.now().isoformat()
                
                cursor.execute('''
                INSERT INTO consciousness_markers 
                (id, session_id, timestamp, consciousness_level, pattern_type, 
                 breath_state, breath_pattern, description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    marker_id, session_id, marker_time, level, "history_point",
                    breathing_data.get("state", "unknown"), breathing_pattern,
                    f"History point {i} in session"
                ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Session {session_id} stored in database")
        return session_id
    
    def get_session(self, session_id: str) -> Dict:
        """
        Retrieve a session by ID
        
        Args:
            session_id: ID of the session to retrieve
            
        Returns:
            Session data dictionary
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get the session
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        session_row = cursor.fetchone()
        
        if not session_row:
            conn.close()
            return {}
        
        # Convert to dictionary
        session_data = dict(session_row)
        
        # Get associated growth events
        cursor.execute("SELECT * FROM growth_events WHERE session_id = ?", (session_id,))
        growth_rows = cursor.fetchall()
        session_data["growth_events"] = [dict(row) for row in growth_rows]
        
        # Get consciousness markers
        cursor.execute("SELECT * FROM consciousness_markers WHERE session_id = ?", (session_id,))
        marker_rows = cursor.fetchall()
        session_data["consciousness_markers"] = [dict(row) for row in marker_rows]
        
        # Parse metadata JSON
        if "metadata" in session_data and session_data["metadata"]:
            session_data["metadata"] = json.loads(session_data["metadata"])
        
        conn.close()
        return session_data
    
    def get_sessions_by_pattern(self, breathing_pattern: str) -> List[Dict]:
        """
        Retrieve all sessions with a specific breathing pattern
        
        Args:
            breathing_pattern: Breathing pattern to filter by
            
        Returns:
            List of session data dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM sessions WHERE breathing_pattern = ?", (breathing_pattern,))
        session_ids = [row["id"] for row in cursor.fetchall()]
        
        conn.close()
        
        return [self.get_session(session_id) for session_id in session_ids]
    
    def get_growth_statistics(self) -> Dict:
        """
        Get statistics on neural growth by breathing pattern
        
        Returns:
            Dictionary with growth statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get total neurons created by pattern
        cursor.execute('''
        SELECT s.breathing_pattern, SUM(g.neurons_created) as total_created,
               SUM(g.neurons_pruned) as total_pruned,
               COUNT(DISTINCT s.id) as session_count
        FROM sessions s
        JOIN growth_events g ON s.id = g.session_id
        GROUP BY s.breathing_pattern
        ''')
        
        pattern_stats = {}
        for row in cursor.fetchall():
            pattern, created, pruned, count = row
            pattern_stats[pattern] = {
                "total_created": created,
                "total_pruned": pruned,
                "net_growth": created - pruned,
                "session_count": count,
                "avg_growth_per_session": (created - pruned) / count if count > 0 else 0
            }
        
        # Get average consciousness by pattern
        cursor.execute('''
        SELECT breathing_pattern, AVG(consciousness_peak) as avg_consciousness
        FROM sessions
        GROUP BY breathing_pattern
        ''')
        
        for row in cursor.fetchall():
            pattern, avg_consciousness = row
            if pattern in pattern_stats:
                pattern_stats[pattern]["avg_consciousness"] = avg_consciousness
        
        conn.close()
        
        return {
            "by_pattern": pattern_stats,
            "total_sessions": sum(stats["session_count"] for stats in pattern_stats.values()),
            "total_created": sum(stats["total_created"] for stats in pattern_stats.values()),
            "total_pruned": sum(stats["total_pruned"] for stats in pattern_stats.values()),
            "net_growth": sum(stats["net_growth"] for stats in pattern_stats.values())
        }
    
    def analyze_consciousness_trends(self) -> Dict:
        """
        Analyze trends in consciousness levels
        
        Returns:
            Dictionary with consciousness trend analysis
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get consciousness peaks by pattern
        cursor.execute('''
        SELECT breathing_pattern, AVG(consciousness_peak) as avg_peak,
               MAX(consciousness_peak) as max_peak,
               MIN(consciousness_peak) as min_peak,
               COUNT(*) as session_count
        FROM sessions
        GROUP BY breathing_pattern
        ''')
        
        consciousness_by_pattern = {}
        for row in cursor.fetchall():
            pattern, avg, max_val, min_val, count = row
            consciousness_by_pattern[pattern] = {
                "average_peak": avg,
                "maximum_peak": max_val,
                "minimum_peak": min_val,
                "session_count": count
            }
        
        # Get patterns with highest consciousness
        cursor.execute('''
        SELECT breathing_pattern, consciousness_peak, id, timestamp
        FROM sessions
        ORDER BY consciousness_peak DESC
        LIMIT 5
        ''')
        
        top_sessions = []
        for row in cursor.fetchall():
            pattern, peak, session_id, timestamp = row
            top_sessions.append({
                "session_id": session_id,
                "breathing_pattern": pattern,
                "consciousness_peak": peak,
                "timestamp": timestamp
            })
        
        conn.close()
        
        return {
            "by_pattern": consciousness_by_pattern,
            "top_sessions": top_sessions,
            "best_pattern": max(
                consciousness_by_pattern.items(), 
                key=lambda x: x[1]["average_peak"]
            )[0] if consciousness_by_pattern else None
        }
    
    def visualize_growth_by_pattern(self, save_path: Optional[str] = None):
        """
        Create visualization of neural growth by breathing pattern
        
        Args:
            save_path: Optional path to save the visualization image
        """
        stats = self.get_growth_statistics()
        pattern_stats = stats["by_pattern"]
        
        if not pattern_stats:
            logger.warning("No growth data available for visualization")
            return
        
        # Prepare data for plotting
        patterns = list(pattern_stats.keys())
        created = [pattern_stats[p]["total_created"] for p in patterns]
        pruned = [pattern_stats[p]["total_pruned"] for p in patterns]
        net_growth = [pattern_stats[p]["net_growth"] for p in patterns]
        
        # Create bar chart
        plt.figure(figsize=(12, 8))
        x = np.arange(len(patterns))
        width = 0.25
        
        plt.bar(x - width, created, width, label='Neurons Created')
        plt.bar(x, pruned, width, label='Neurons Pruned')
        plt.bar(x + width, net_growth, width, label='Net Growth')
        
        plt.xlabel('Breathing Pattern')
        plt.ylabel('Neuron Count')
        plt.title('Neural Growth by Breathing Pattern')
        plt.xticks(x, patterns)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Growth visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_consciousness_trends(self, save_path: Optional[str] = None):
        """
        Create visualization of consciousness trends by breathing pattern
        
        Args:
            save_path: Optional path to save the visualization image
        """
        trends = self.analyze_consciousness_trends()
        pattern_data = trends["by_pattern"]
        
        if not pattern_data:
            logger.warning("No consciousness data available for visualization")
            return
        
        # Prepare data for plotting
        patterns = list(pattern_data.keys())
        avg_peaks = [pattern_data[p]["average_peak"] for p in patterns]
        max_peaks = [pattern_data[p]["maximum_peak"] for p in patterns]
        min_peaks = [pattern_data[p]["minimum_peak"] for p in patterns]
        
        # Create bar chart
        plt.figure(figsize=(12, 8))
        x = np.arange(len(patterns))
        width = 0.25
        
        plt.bar(x - width, avg_peaks, width, label='Average Peak')
        plt.bar(x, max_peaks, width, label='Maximum Peak')
        plt.bar(x + width, min_peaks, width, label='Minimum Peak')
        
        plt.xlabel('Breathing Pattern')
        plt.ylabel('Consciousness Level')
        plt.title('Consciousness Metrics by Breathing Pattern')
        plt.xticks(x, patterns)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Consciousness visualization saved to {save_path}")
        
        plt.show()
    
    def export_data(self, export_path: str, format: str = "json"):
        """
        Export database data to JSON or CSV format
        
        Args:
            export_path: Directory to export data to
            format: Export format ("json" or "csv")
        """
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Export each table
        tables = ["sessions", "growth_events", "consciousness_markers", "neural_metrics"]
        
        for table in tables:
            cursor.execute(f"SELECT * FROM {table}")
            rows = cursor.fetchall()
            data = [dict(row) for row in rows]
            
            if format == "json":
                with open(export_dir / f"{table}.json", "w") as f:
                    json.dump(data, f, indent=2)
            elif format == "csv":
                if not data:
                    continue
                    
                df = pd.DataFrame(data)
                df.to_csv(export_dir / f"{table}.csv", index=False)
        
        conn.close()
        logger.info(f"Database exported to {export_path} in {format} format")
    
    def import_data(self, import_path: str, format: str = "json"):
        """
        Import data from JSON or CSV files
        
        Args:
            import_path: Directory to import data from
            format: Import format ("json" or "csv")
        """
        import_dir = Path(import_path)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Import each table
        tables = ["sessions", "growth_events", "consciousness_markers", "neural_metrics"]
        
        for table in tables:
            file_path = import_dir / f"{table}.{format}"
            if not file_path.exists():
                logger.warning(f"File {file_path} not found, skipping")
                continue
            
            if format == "json":
                with open(file_path, "r") as f:
                    data = json.load(f)
                
                if not data:
                    continue
                
                # Get column names from the first item
                columns = list(data[0].keys())
                placeholders = ", ".join(["?"] * len(columns))
                columns_str = ", ".join(columns)
                
                # Insert data
                for item in data:
                    values = [item[col] for col in columns]
                    cursor.execute(f"INSERT OR REPLACE INTO {table} ({columns_str}) VALUES ({placeholders})", values)
            
            elif format == "csv":
                df = pd.read_csv(file_path)
                
                # Convert DataFrame to list of dictionaries
                data = df.to_dict(orient="records")
                
                if not data:
                    continue
                
                # Get column names
                columns = list(data[0].keys())
                placeholders = ", ".join(["?"] * len(columns))
                columns_str = ", ".join(columns)
                
                # Insert data
                for item in data:
                    values = [item[col] for col in columns]
                    cursor.execute(f"INSERT OR REPLACE INTO {table} ({columns_str}) VALUES ({placeholders})", values)
        
        conn.commit()
        conn.close()
        logger.info(f"Data imported from {import_path} in {format} format")

# Integration with IntegratedNeuralPlayground
def integrate_with_neural_playground(playground, db_path: str = "neural_database.db"):
    """
    Integrate database with the neural playground
    
    Args:
        playground: IntegratedNeuralPlayground instance
        db_path: Path to the database file
        
    Returns:
        NeuralDatabase instance
    """
    db = NeuralDatabase(db_path)
    
    # Define hook for storing session data
    def post_play_hook(playground, play_result):
        """Hook called after play session to store data"""
        try:
            # Get state data
            pre_state = playground.get_current_state() 
            neural_state = pre_state["neural"]
            size = pre_state["size"]
            
            # Add network size info
            if not "network_size" in play_result:
                play_result["network_size"] = {
                    "initial": size - play_result.get("brain_growth", {}).get("neurons_created", 0),
                    "final": size
                }
            
            # Store session data
            db.store_session(play_result)
            
        except Exception as e:
            logger.error(f"Error in database post-play hook: {e}")
    
    # Register the hook if possible
    if hasattr(playground, 'integration') and hasattr(playground.integration, 'registered_components'):
        playground.integration["registered_components"]["neural_database"] = {
            "component": db,
            "hooks": {
                "post_play": post_play_hook
            }
        }
    
    logger.info("Neural database integrated with playground")
    return db

if __name__ == "__main__":
    # Example usage
    db = NeuralDatabase()
    
    # Create a test session
    test_session = {
        "session_id": f"test_{int(time.time())}",
        "duration": 100,
        "play_type": "mixed",
        "intensity": 0.7,
        "consciousness_peak": 0.85,
        "patterns_detected": 12,
        "total_activations": 4250,
        "network_size": {
            "initial": 100,
            "final": 135
        },
        "breathing_data": {
            "pattern": "meditative",
            "coherence": 0.92,
            "rate": 6.0,
            "state": "inhale",
            "amplitude": 0.75
        },
        "brain_growth": {
            "growth_state": "EXPANSION",
            "neurons_created": 40,
            "neurons_pruned": 5,
            "regions_formed": 2
        },
        "consciousness_history": [0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.8, 0.75]
    }
    
    session_id = db.store_session(test_session)
    print(f"Stored test session with ID: {session_id}")
    
    # Visualize data
    db.visualize_growth_by_pattern()
    db.visualize_consciousness_trends() 