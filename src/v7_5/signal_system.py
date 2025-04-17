#!/usr/bin/env python3
"""
LUMINA v7.5 Signal System
Core component that manages signal communication between components
"""

import os
import logging
import asyncio
from typing import Dict, List, Any, Callable, Optional
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", f"signal_system_{os.getpid()}.log"))
    ]
)
logger = logging.getLogger("SignalSystem")

class SignalComponent:
    """Base class for components that use the signal system"""
    
    def __init__(self, name: str, signal_bus: 'SignalBus'):
        self.name = name
        self.signal_bus = signal_bus
        self.handlers: Dict[str, Callable] = {}
        self.logger = logging.getLogger(f"SignalComponent.{name}")
        self._initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the signal component"""
        try:
            if self._initialized:
                return True
                
            # Register with signal bus
            self.signal_bus.register_component(self.name, self)
            self._initialized = True
            self.logger.info(f"Signal component {self.name} initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize signal component {self.name}: {e}")
            return False
        
    def register_handler(self, signal_type: str, handler: Callable) -> None:
        """Register a handler for a signal type"""
        self.handlers[signal_type] = handler
        self.signal_bus.register_handler(signal_type, handler)
        self.logger.debug(f"Registered handler for signal type: {signal_type}")
        
    def unregister_handler(self, signal_type: str) -> None:
        """Unregister a handler for a signal type"""
        if signal_type in self.handlers:
            self.signal_bus.unregister_handler(signal_type, self.handlers[signal_type])
            del self.handlers[signal_type]
            self.logger.debug(f"Unregistered handler for signal type: {signal_type}")
            
    def unregister_all_handlers(self) -> None:
        """Unregister all handlers"""
        for signal_type in list(self.handlers.keys()):
            self.unregister_handler(signal_type)
        self.logger.debug("Unregistered all handlers")
            
    async def emit_signal(self, signal_type: str, data: Any) -> None:
        """Emit a signal through the signal bus"""
        if not self._initialized:
            self.logger.warning(f"Cannot emit signal: {self.name} is not connected")
            return
            
        await self.signal_bus.emit(signal_type, data)
        self.logger.debug(f"Emitted signal: {signal_type}")
        
    async def cleanup(self) -> None:
        """Clean up component resources"""
        self.unregister_all_handlers()
        if self._initialized:
            self.signal_bus.unregister_component(self.name)
            self._initialized = False
        self.logger.info(f"Component {self.name} cleaned up")

class SignalBus:
    """Signal bus for managing signal communication between components"""
    
    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.components: Dict[str, Any] = {}
        
    async def initialize(self) -> bool:
        """Initialize the signal bus"""
        try:
            if self._running:
                return True
                
            self._running = True
            self._task = asyncio.create_task(self._process_queue())
            logger.info("Signal bus initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize signal bus: {e}")
            return False
            
    def register_handler(self, signal_type: str, handler: Callable) -> None:
        """Register a handler for a signal type"""
        self._handlers[signal_type].append(handler)
        logger.debug(f"Registered handler for signal type: {signal_type}")
        
    def unregister_handler(self, signal_type: str, handler: Callable) -> None:
        """Unregister a handler for a signal type"""
        if signal_type in self._handlers:
            self._handlers[signal_type].remove(handler)
            logger.debug(f"Unregistered handler for signal type: {signal_type}")
            
    async def emit(self, signal_type: str, data: Any) -> None:
        """Emit a signal to all registered handlers"""
        await self._queue.put((signal_type, data))
        logger.debug(f"Emitted signal: {signal_type}")
        
    def register_component(self, name: str, component: Any) -> None:
        """Register a component with the signal bus"""
        self.components[name] = component
        logger.debug(f"Registered component: {name}")
        
    def unregister_component(self, name: str) -> None:
        """Unregister a component from the signal bus"""
        if name in self.components:
            del self.components[name]
            logger.debug(f"Unregistered component: {name}")
            
    def get_component(self, name: str) -> Optional[Any]:
        """Get a registered component by name"""
        return self.components.get(name)
        
    async def broadcast(self, signal_type: str, data: Any) -> None:
        """Broadcast a signal to all registered handlers"""
        await self.emit(signal_type, data)
        
    async def _process_queue(self) -> None:
        """Process signals from the queue"""
        while self._running:
            try:
                signal_type, data = await self._queue.get()
                
                if signal_type in self._handlers:
                    for handler in self._handlers[signal_type]:
                        try:
                            await handler(data)
                        except Exception as e:
                            logger.error(f"Error in signal handler for {signal_type}: {e}")
                            
                self._queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing signal queue: {e}")
                
    async def cleanup(self) -> None:
        """Clean up signal bus resources"""
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
                
        self._handlers.clear()
        self.components.clear()
        logger.info("Signal bus cleaned up") 