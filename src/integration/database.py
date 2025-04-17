#!/usr/bin/env python3
"""
Database Module for Signal Storage

This module handles the storage and retrieval of signals in the SQLite database.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "data" / "signals.db"

def init_db() -> None:
    """Initialize the database with required tables"""
    try:
        # Create data directory if it doesn't exist
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Create signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    gate_id TEXT NOT NULL,
                    gate_type TEXT NOT NULL,
                    output REAL NOT NULL,
                    inputs TEXT NOT NULL,
                    connections TEXT NOT NULL,
                    visual_effects TEXT NOT NULL,
                    path_type TEXT NOT NULL,
                    gate_state TEXT NOT NULL
                )
            ''')
            
            # Create metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    gate_id TEXT NOT NULL,
                    health_score REAL NOT NULL,
                    latency REAL NOT NULL,
                    load REAL NOT NULL,
                    memory REAL NOT NULL,
                    success_rate REAL NOT NULL
                )
            ''')
            
            # Create patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    gate_id TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    pattern_data TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            logger.info("Database initialized successfully")
            
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

def store_signal(
    gate_id: str,
    gate_type: str,
    output: float,
    inputs: Dict[str, float],
    connections: List[str],
    visual_effects: Dict[str, Any],
    path_type: str,
    gate_state: str
) -> None:
    """Store a signal in the database"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO signals (
                    timestamp, gate_id, gate_type, output, inputs,
                    connections, visual_effects, path_type, gate_state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                gate_id,
                gate_type,
                output,
                json.dumps(inputs),
                json.dumps(connections),
                json.dumps(visual_effects),
                path_type,
                gate_state
            ))
            
            conn.commit()
            logger.debug(f"Stored signal for gate {gate_id}")
            
    except Exception as e:
        logger.error(f"Error storing signal: {e}")
        raise

def store_metrics(
    gate_id: str,
    health_score: float,
    latency: float,
    load: float,
    memory: float,
    success_rate: float
) -> None:
    """Store metrics in the database"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO metrics (
                    timestamp, gate_id, health_score, latency,
                    load, memory, success_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                gate_id,
                health_score,
                latency,
                load,
                memory,
                success_rate
            ))
            
            conn.commit()
            logger.debug(f"Stored metrics for gate {gate_id}")
            
    except Exception as e:
        logger.error(f"Error storing metrics: {e}")
        raise

def store_pattern(
    gate_id: str,
    pattern_type: str,
    pattern_data: Dict[str, Any]
) -> None:
    """Store a pattern in the database"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO patterns (
                    timestamp, gate_id, pattern_type, pattern_data
                ) VALUES (?, ?, ?, ?)
            ''', (
                datetime.now(),
                gate_id,
                pattern_type,
                json.dumps(pattern_data)
            ))
            
            conn.commit()
            logger.debug(f"Stored pattern for gate {gate_id}")
            
    except Exception as e:
        logger.error(f"Error storing pattern: {e}")
        raise

def get_recent_signals(
    gate_id: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """Get recent signals from the database"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            if gate_id:
                cursor.execute('''
                    SELECT * FROM signals
                    WHERE gate_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (gate_id, limit))
            else:
                cursor.execute('''
                    SELECT * FROM signals
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (limit,))
                
            signals = []
            for row in cursor.fetchall():
                signals.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'gate_id': row[2],
                    'gate_type': row[3],
                    'output': row[4],
                    'inputs': json.loads(row[5]),
                    'connections': json.loads(row[6]),
                    'visual_effects': json.loads(row[7]),
                    'path_type': row[8],
                    'gate_state': row[9]
                })
                
            return signals
            
    except Exception as e:
        logger.error(f"Error getting recent signals: {e}")
        raise

def get_gate_metrics(
    gate_id: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """Get metrics for a specific gate"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            if start_time and end_time:
                cursor.execute('''
                    SELECT * FROM metrics
                    WHERE gate_id = ? AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                ''', (gate_id, start_time, end_time))
            else:
                cursor.execute('''
                    SELECT * FROM metrics
                    WHERE gate_id = ?
                    ORDER BY timestamp
                ''', (gate_id,))
                
            metrics = []
            for row in cursor.fetchall():
                metrics.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'gate_id': row[2],
                    'health_score': row[3],
                    'latency': row[4],
                    'load': row[5],
                    'memory': row[6],
                    'success_rate': row[7]
                })
                
            return metrics
            
    except Exception as e:
        logger.error(f"Error getting gate metrics: {e}")
        raise

def get_patterns(
    gate_id: str,
    pattern_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get patterns for a specific gate"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            if pattern_type:
                cursor.execute('''
                    SELECT * FROM patterns
                    WHERE gate_id = ? AND pattern_type = ?
                    ORDER BY timestamp
                ''', (gate_id, pattern_type))
            else:
                cursor.execute('''
                    SELECT * FROM patterns
                    WHERE gate_id = ?
                    ORDER BY timestamp
                ''', (gate_id,))
                
            patterns = []
            for row in cursor.fetchall():
                patterns.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'gate_id': row[2],
                    'pattern_type': row[3],
                    'pattern_data': json.loads(row[4])
                })
                
            return patterns
            
    except Exception as e:
        logger.error(f"Error getting patterns: {e}")
        raise 