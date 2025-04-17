"""
Data Flow Manager for handling information flow between Spiderweb and the V1 database.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from .bridge_connector import BridgeConnector

logger = logging.getLogger(__name__)

class DataFlowManager:
    """Manages data flow between Spiderweb and the V1 database."""
    
    def __init__(self, bridge_connector: BridgeConnector):
        """
        Initialize the data flow manager.
        
        Args:
            bridge_connector: Instance of BridgeConnector for database operations
        """
        self.connector = bridge_connector
        self.event_handlers = {}
        self.metric_buffers = {}
        self.sync_queue = asyncio.Queue()
        self._setup_handlers()
        
    def _setup_handlers(self):
        """Set up event handlers for different types of data."""
        self.event_handlers = {
            'node_created': self._handle_node_created,
            'node_updated': self._handle_node_updated,
            'node_deleted': self._handle_node_deleted,
            'connection_created': self._handle_connection_created,
            'connection_updated': self._handle_connection_updated,
            'connection_deleted': self._handle_connection_deleted,
            'quantum_sync': self._handle_quantum_sync,
            'cosmic_sync': self._handle_cosmic_sync,
            'metric_update': self._handle_metric_update,
            'state_change': self._handle_state_change,
            'error_occurred': self._handle_error
        }
        
    async def start(self):
        """Start the data flow manager."""
        try:
            # Start background tasks
            self.sync_task = asyncio.create_task(self._sync_loop())
            self.metric_task = asyncio.create_task(self._metric_processing_loop())
            logger.info("Data flow manager started successfully")
        except Exception as e:
            logger.error(f"Error starting data flow manager: {str(e)}")
            raise
            
    async def stop(self):
        """Stop the data flow manager."""
        try:
            # Cancel background tasks
            if hasattr(self, 'sync_task'):
                self.sync_task.cancel()
            if hasattr(self, 'metric_task'):
                self.metric_task.cancel()
            logger.info("Data flow manager stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping data flow manager: {str(e)}")
            
    async def handle_event(self, event_type: str, data: Dict[str, Any]):
        """
        Handle an event from the Spiderweb system.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        try:
            if event_type in self.event_handlers:
                await self.event_handlers[event_type](data)
            else:
                logger.warning(f"No handler for event type: {event_type}")
        except Exception as e:
            logger.error(f"Error handling event {event_type}: {str(e)}")
            await self._handle_error({
                'event_type': event_type,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            
    async def _handle_node_created(self, data: Dict[str, Any]):
        """Handle node creation event."""
        try:
            node_data = {
                'node_id': data['id'],
                'name': data.get('name', f"Node_{data['id']}"),
                'type': data.get('type', 'unknown'),
                'status': data.get('status', 'CREATED'),
                'version': data.get('version', 'v1'),
                'config': data.get('config', {}),
                'metadata': data.get('metadata', {})
            }
            
            if self.connector.handle_node_creation(node_data):
                await self.sync_queue.put({
                    'event_type': 'node_created',
                    'status': 'success',
                    'data': node_data
                })
                
        except Exception as e:
            logger.error(f"Error handling node creation: {str(e)}")
            await self._handle_error({
                'event_type': 'node_created',
                'error': str(e),
                'data': data
            })
            
    async def _handle_node_updated(self, data: Dict[str, Any]):
        """Handle node update event."""
        try:
            if 'id' in data and 'status' in data:
                if self.connector.handle_node_status_change(data['id'], data['status']):
                    await self.sync_queue.put({
                        'event_type': 'node_updated',
                        'status': 'success',
                        'data': data
                    })
                    
        except Exception as e:
            logger.error(f"Error handling node update: {str(e)}")
            await self._handle_error({
                'event_type': 'node_updated',
                'error': str(e),
                'data': data
            })
            
    async def _handle_node_deleted(self, data: Dict[str, Any]):
        """Handle node deletion event."""
        try:
            if 'id' in data:
                # Mark node as deleted in database
                if self.connector.handle_node_status_change(data['id'], 'DELETED'):
                    await self.sync_queue.put({
                        'event_type': 'node_deleted',
                        'status': 'success',
                        'data': data
                    })
                    
        except Exception as e:
            logger.error(f"Error handling node deletion: {str(e)}")
            await self._handle_error({
                'event_type': 'node_deleted',
                'error': str(e),
                'data': data
            })
            
    async def _handle_connection_created(self, data: Dict[str, Any]):
        """Handle connection creation event."""
        try:
            connection_data = {
                'source_id': data['source'],
                'target_id': data['target'],
                'connection_type': data.get('type', 'standard'),
                'strength': data.get('strength', 1.0),
                'status': data.get('status', 'ACTIVE'),
                'metadata': data.get('metadata', {})
            }
            
            if self.connector.handle_connection_creation(connection_data):
                await self.sync_queue.put({
                    'event_type': 'connection_created',
                    'status': 'success',
                    'data': connection_data
                })
                
        except Exception as e:
            logger.error(f"Error handling connection creation: {str(e)}")
            await self._handle_error({
                'event_type': 'connection_created',
                'error': str(e),
                'data': data
            })
            
    async def _handle_connection_updated(self, data: Dict[str, Any]):
        """Handle connection update event."""
        try:
            if 'id' in data and 'strength' in data:
                if self.connector.handle_connection_strength_update(
                    data['id'], data['strength']
                ):
                    await self.sync_queue.put({
                        'event_type': 'connection_updated',
                        'status': 'success',
                        'data': data
                    })
                    
        except Exception as e:
            logger.error(f"Error handling connection update: {str(e)}")
            await self._handle_error({
                'event_type': 'connection_updated',
                'error': str(e),
                'data': data
            })
            
    async def _handle_quantum_sync(self, data: Dict[str, Any]):
        """Handle quantum synchronization event."""
        try:
            metric_data = {
                'metric_type': 'quantum_sync',
                'value': data.get('field_strength', 0.0),
                'node_id': data.get('node_id'),
                'metadata': {
                    'entanglement_count': data.get('entangled_nodes_count', 0),
                    'phase': data.get('phase', 0.0),
                    'frequency': data.get('frequency', 0.0)
                }
            }
            
            if self.connector.handle_metric_update(metric_data):
                await self.sync_queue.put({
                    'event_type': 'quantum_sync',
                    'status': 'success',
                    'data': metric_data
                })
                
        except Exception as e:
            logger.error(f"Error handling quantum sync: {str(e)}")
            await self._handle_error({
                'event_type': 'quantum_sync',
                'error': str(e),
                'data': data
            })
            
    async def _handle_cosmic_sync(self, data: Dict[str, Any]):
        """Handle cosmic synchronization event."""
        try:
            metric_data = {
                'metric_type': 'cosmic_sync',
                'value': data.get('field_strength', 0.0),
                'node_id': data.get('node_id'),
                'metadata': {
                    'dimensional_resonance': data.get('resonance', 0.0),
                    'universal_phase': data.get('phase', 0.0),
                    'cosmic_frequency': data.get('frequency', 0.0)
                }
            }
            
            if self.connector.handle_metric_update(metric_data):
                await self.sync_queue.put({
                    'event_type': 'cosmic_sync',
                    'status': 'success',
                    'data': metric_data
                })
                
        except Exception as e:
            logger.error(f"Error handling cosmic sync: {str(e)}")
            await self._handle_error({
                'event_type': 'cosmic_sync',
                'error': str(e),
                'data': data
            })
            
    async def _handle_metric_update(self, data: Dict[str, Any]):
        """Handle metric update event."""
        try:
            node_id = data.get('node_id')
            if node_id not in self.metric_buffers:
                self.metric_buffers[node_id] = []
                
            self.metric_buffers[node_id].append(data)
            
            # Process buffer if it reaches threshold or is time-critical
            if len(self.metric_buffers[node_id]) >= 10 or data.get('critical', False):
                await self._process_metric_buffer(node_id)
                
        except Exception as e:
            logger.error(f"Error handling metric update: {str(e)}")
            await self._handle_error({
                'event_type': 'metric_update',
                'error': str(e),
                'data': data
            })
            
    async def _handle_state_change(self, data: Dict[str, Any]):
        """Handle state change event."""
        try:
            # Log state change event
            self.connector.db.add_sync_event({
                'event_type': 'state_change',
                'status': 'success',
                'source_version': data.get('version', 'v1'),
                'target_version': 'v1',
                'details': data
            })
            
            await self.sync_queue.put({
                'event_type': 'state_change',
                'status': 'success',
                'data': data
            })
            
        except Exception as e:
            logger.error(f"Error handling state change: {str(e)}")
            await self._handle_error({
                'event_type': 'state_change',
                'error': str(e),
                'data': data
            })
            
    async def _handle_error(self, error_data: Dict[str, Any]):
        """Handle error event."""
        try:
            # Log error event
            self.connector.db.add_sync_event({
                'event_type': 'error',
                'status': 'error',
                'source_version': error_data.get('version', 'v1'),
                'target_version': 'v1',
                'details': error_data,
                'error_message': error_data.get('error', 'Unknown error')
            })
            
        except Exception as e:
            logger.error(f"Error handling error event: {str(e)}")
            
    async def _sync_loop(self):
        """Background task for processing sync queue."""
        try:
            while True:
                event = await self.sync_queue.get()
                try:
                    # Process sync event
                    self.connector.db.add_sync_event({
                        'event_type': event['event_type'],
                        'status': event['status'],
                        'source_version': 'v1',
                        'target_version': 'v1',
                        'details': event['data']
                    })
                except Exception as e:
                    logger.error(f"Error processing sync event: {str(e)}")
                finally:
                    self.sync_queue.task_done()
                    
        except asyncio.CancelledError:
            logger.info("Sync loop cancelled")
        except Exception as e:
            logger.error(f"Error in sync loop: {str(e)}")
            
    async def _metric_processing_loop(self):
        """Background task for processing metric buffers."""
        try:
            while True:
                # Process all metric buffers
                for node_id in list(self.metric_buffers.keys()):
                    await self._process_metric_buffer(node_id)
                await asyncio.sleep(1)  # Check every second
                
        except asyncio.CancelledError:
            logger.info("Metric processing loop cancelled")
        except Exception as e:
            logger.error(f"Error in metric processing loop: {str(e)}")
            
    async def _process_metric_buffer(self, node_id: str):
        """Process metrics in buffer for a node."""
        try:
            if node_id in self.metric_buffers and self.metric_buffers[node_id]:
                metrics = self.metric_buffers[node_id]
                self.metric_buffers[node_id] = []
                
                for metric in metrics:
                    if self.connector.handle_metric_update(metric):
                        await self.sync_queue.put({
                            'event_type': 'metric_update',
                            'status': 'success',
                            'data': metric
                        })
                        
        except Exception as e:
            logger.error(f"Error processing metric buffer for node {node_id}: {str(e)}")
            await self._handle_error({
                'event_type': 'metric_processing',
                'error': str(e),
                'node_id': node_id
            }) 