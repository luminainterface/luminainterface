"""
Centralized Error Handling System

This module provides a comprehensive error handling framework for the Lumina system,
including error categorization, standardized error responses, logging, and recovery mechanisms.
"""

import logging
import time
import traceback
import threading
import uuid
import sys
import json
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Callable, Type, Union, Tuple
from pathlib import Path
from datetime import datetime
from functools import wraps

logger = logging.getLogger(__name__)

# Ensure log directory exists
log_dir = Path("logs/errors")
log_dir.mkdir(parents=True, exist_ok=True)

class ErrorSeverity(Enum):
    """Error severity levels"""
    CRITICAL = 50  # System cannot continue operation
    ERROR = 40     # Operation failed, but system can continue
    WARNING = 30   # Potential issue, operation completed
    INFO = 20      # Informational error, no impact
    DEBUG = 10     # Debug-level issue
    
class ErrorCategory(Enum):
    """Categories of errors for classification"""
    CONFIGURATION = auto()  # Configuration-related errors
    DATABASE = auto()       # Database connectivity or query errors
    NETWORK = auto()        # Network/communication errors
    API = auto()            # API-related errors
    AUTHENTICATION = auto() # Auth-related errors
    PERMISSION = auto()     # Permission-related errors
    VALIDATION = auto()     # Data validation errors
    PROCESSING = auto()     # Data processing errors
    RESOURCE = auto()       # Resource unavailable/exhausted
    COMPATIBILITY = auto()  # Version compatibility errors
    INTEGRATION = auto()    # Integration with other components
    INTERNAL = auto()       # Internal system errors
    UNKNOWN = auto()        # Uncategorized errors
    
class ErrorResponse:
    """Standardized error response structure"""
    
    def __init__(self, 
                message: str,
                category: ErrorCategory,
                severity: ErrorSeverity,
                error_code: str = None,
                details: Dict[str, Any] = None,
                exception: Exception = None,
                component: str = None,
                retry_allowed: bool = True,
                recovery_hint: str = None):
        """
        Initialize an error response.
        
        Args:
            message: Human-readable error message
            category: Error category
            severity: Error severity
            error_code: System-specific error code
            details: Additional error details
            exception: Original exception if applicable
            component: Component where the error occurred
            retry_allowed: Whether retry is allowed for this error
            recovery_hint: Hint for recovery actions
        """
        self.message = message
        self.category = category
        self.severity = severity
        self.error_code = error_code or f"{category.name}_{uuid.uuid4().hex[:8]}"
        self.details = details or {}
        self.exception = exception
        self.component = component
        self.timestamp = datetime.now().isoformat()
        self.retry_allowed = retry_allowed
        self.recovery_hint = recovery_hint
        
        # Extract traceback if exception provided
        if exception:
            self.details["exception_type"] = type(exception).__name__
            self.details["exception_message"] = str(exception)
            self.details["traceback"] = traceback.format_exc()
            
        # Log the error
        self._log_error()
            
    def _log_error(self):
        """Log the error to the appropriate destinations"""
        log_message = f"[{self.error_code}] {self.message}"
        
        # Add details for debug logging
        if self.details:
            detail_str = " | ".join(f"{k}={v}" for k, v in self.details.items() 
                                  if k != "traceback")
            log_message += f" | {detail_str}"
            
        # Log to the appropriate level
        if self.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message, exc_info=self.exception)
        elif self.severity == ErrorSeverity.ERROR:
            logger.error(log_message, exc_info=self.exception)
        elif self.severity == ErrorSeverity.WARNING:
            logger.warning(log_message)
        elif self.severity == ErrorSeverity.INFO:
            logger.info(log_message)
        elif self.severity == ErrorSeverity.DEBUG:
            logger.debug(log_message)
            
        # For CRITICAL and ERROR, also log to error file
        if self.severity in (ErrorSeverity.CRITICAL, ErrorSeverity.ERROR):
            self._write_to_error_log()
            
    def _write_to_error_log(self):
        """Write detailed error information to log file"""
        try:
            # Generate log filename with timestamp
            date_str = datetime.now().strftime("%Y%m%d")
            log_file = log_dir / f"error_log_{date_str}.json"
            
            # Create log entry
            log_entry = {
                "error_code": self.error_code,
                "message": self.message,
                "category": self.category.name,
                "severity": self.severity.name,
                "component": self.component,
                "timestamp": self.timestamp,
                "details": self.details
            }
            
            # Append to log file
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            logger.error(f"Failed to write to error log: {str(e)}")
            
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert error response to dictionary.
        
        Returns:
            Dictionary representation of the error
        """
        return {
            "error": True,
            "error_code": self.error_code,
            "message": self.message,
            "category": self.category.name,
            "severity": self.severity.name,
            "component": self.component,
            "timestamp": self.timestamp,
            "retry_allowed": self.retry_allowed,
            "recovery_hint": self.recovery_hint
        }
        
    def to_user_friendly_dict(self) -> Dict[str, Any]:
        """
        Convert to a user-friendly dictionary without technical details.
        
        Returns:
            User-friendly dictionary representation
        """
        return {
            "error": True,
            "message": self.message,
            "error_code": self.error_code,
            "timestamp": self.timestamp,
            "recovery_hint": self.recovery_hint
        }
        
    def __str__(self) -> str:
        """String representation of the error"""
        return f"[{self.error_code}] {self.message} ({self.category.name}/{self.severity.name})"


class LuminaError(Exception):
    """Base exception class for Lumina-specific errors"""
    
    def __init__(self, 
                message: str,
                category: ErrorCategory = ErrorCategory.UNKNOWN,
                severity: ErrorSeverity = ErrorSeverity.ERROR,
                error_code: str = None,
                details: Dict[str, Any] = None,
                component: str = None,
                retry_allowed: bool = True,
                recovery_hint: str = None):
        """
        Initialize a Lumina-specific error.
        
        Args:
            message: Human-readable error message
            category: Error category
            severity: Error severity
            error_code: System-specific error code
            details: Additional error details
            component: Component where the error occurred
            retry_allowed: Whether retry is allowed for this error
            recovery_hint: Hint for recovery actions
        """
        self.error_response = ErrorResponse(
            message=message,
            category=category,
            severity=severity,
            error_code=error_code,
            details=details,
            component=component,
            retry_allowed=retry_allowed,
            recovery_hint=recovery_hint
        )
        
        super().__init__(message)
        
    def __str__(self) -> str:
        """String representation of the error"""
        return str(self.error_response)
        

# Common error types
class ConfigurationError(LuminaError):
    """Error related to system configuration"""
    def __init__(self, message, **kwargs):
        super().__init__(message, category=ErrorCategory.CONFIGURATION, **kwargs)
        
class DatabaseError(LuminaError):
    """Error related to database operations"""
    def __init__(self, message, **kwargs):
        super().__init__(message, category=ErrorCategory.DATABASE, **kwargs)
        
class NetworkError(LuminaError):
    """Error related to network operations"""
    def __init__(self, message, **kwargs):
        super().__init__(message, category=ErrorCategory.NETWORK, **kwargs)
        
class APIError(LuminaError):
    """Error related to API operations"""
    def __init__(self, message, **kwargs):
        super().__init__(message, category=ErrorCategory.API, **kwargs)
        
class ValidationError(LuminaError):
    """Error related to data validation"""
    def __init__(self, message, **kwargs):
        super().__init__(message, category=ErrorCategory.VALIDATION, 
                       severity=ErrorSeverity.WARNING, **kwargs)
        
class AuthenticationError(LuminaError):
    """Error related to authentication"""
    def __init__(self, message, **kwargs):
        super().__init__(message, category=ErrorCategory.AUTHENTICATION, **kwargs)
        
class PermissionError(LuminaError):
    """Error related to permissions"""
    def __init__(self, message, **kwargs):
        super().__init__(message, category=ErrorCategory.PERMISSION, **kwargs)
        
class ResourceError(LuminaError):
    """Error related to resource availability"""
    def __init__(self, message, **kwargs):
        super().__init__(message, category=ErrorCategory.RESOURCE, **kwargs)
        
class CompatibilityError(LuminaError):
    """Error related to version compatibility"""
    def __init__(self, message, **kwargs):
        super().__init__(message, category=ErrorCategory.COMPATIBILITY, **kwargs)
        
class IntegrationError(LuminaError):
    """Error related to component integration"""
    def __init__(self, message, **kwargs):
        super().__init__(message, category=ErrorCategory.INTEGRATION, **kwargs)


class ErrorManager:
    """Centralized error management system"""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton implementation"""
        if cls._instance is None:
            cls._instance = super(ErrorManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the error manager"""
        if self._initialized:
            return
            
        self._initialized = True
        self._error_handlers = {}
        self._global_handlers = []
        self._errors = []  # Recent errors
        self._max_errors = 1000  # Maximum number of errors to keep
        self._lock = threading.RLock()
        
        # Error recovery strategies
        self._recovery_strategies = {}
        
        logger.info("ErrorManager initialized")
        
    def register_handler(self, 
                        handler: Callable[[ErrorResponse], None],
                        categories: List[ErrorCategory] = None,
                        severities: List[ErrorSeverity] = None) -> None:
        """
        Register an error handler function.
        
        Args:
            handler: Function to handle errors
            categories: Categories to handle (None for all)
            severities: Severities to handle (None for all)
        """
        with self._lock:
            if categories is None and severities is None:
                self._global_handlers.append(handler)
                logger.debug(f"Registered global error handler: {handler.__name__}")
            else:
                key = (tuple(categories) if categories else None, 
                      tuple(severities) if severities else None)
                      
                if key not in self._error_handlers:
                    self._error_handlers[key] = []
                    
                self._error_handlers[key].append(handler)
                logger.debug(f"Registered error handler for {key}: {handler.__name__}")
                
    def register_recovery_strategy(self, 
                                 category: ErrorCategory,
                                 strategy: Callable[[ErrorResponse], bool]) -> None:
        """
        Register a recovery strategy for a category of errors.
        
        Args:
            category: Error category to handle
            strategy: Function that attempts recovery, returns success boolean
        """
        with self._lock:
            if category not in self._recovery_strategies:
                self._recovery_strategies[category] = []
                
            self._recovery_strategies[category].append(strategy)
            logger.debug(f"Registered recovery strategy for {category.name}")
                
    def handle_error(self, 
                    error: Union[Exception, ErrorResponse],
                    component: str = None) -> ErrorResponse:
        """
        Process an error through the error management system.
        
        Args:
            error: Exception or ErrorResponse to handle
            component: Component where the error occurred
            
        Returns:
            ErrorResponse instance
        """
        # Convert to ErrorResponse if needed
        if isinstance(error, LuminaError):
            error_response = error.error_response
        elif isinstance(error, Exception) and not isinstance(error, ErrorResponse):
            error_response = ErrorResponse(
                message=str(error),
                category=ErrorCategory.UNKNOWN,
                severity=ErrorSeverity.ERROR,
                exception=error,
                component=component
            )
        else:
            error_response = error
            
        # Store in recent errors
        with self._lock:
            self._errors.append(error_response)
            if len(self._errors) > self._max_errors:
                self._errors.pop(0)
                
        # Call appropriate handlers
        self._call_handlers(error_response)
        
        # Attempt recovery if allowed
        if error_response.retry_allowed:
            self._attempt_recovery(error_response)
                
        return error_response
        
    def _call_handlers(self, error: ErrorResponse) -> None:
        """Call appropriate error handlers for an error"""
        with self._lock:
            # Call global handlers
            for handler in self._global_handlers:
                try:
                    handler(error)
                except Exception as e:
                    logger.error(f"Error in global handler {handler.__name__}: {str(e)}")
                    
            # Call specific handlers
            for (categories, severities), handlers in self._error_handlers.items():
                # Check if this handler applies
                if ((categories is None or error.category in categories) and 
                    (severities is None or error.severity in severities)):
                    
                    for handler in handlers:
                        try:
                            handler(error)
                        except Exception as e:
                            logger.error(f"Error in handler {handler.__name__}: {str(e)}")
                            
    def _attempt_recovery(self, error: ErrorResponse) -> bool:
        """
        Attempt to recover from an error.
        
        Args:
            error: ErrorResponse to recover from
            
        Returns:
            Boolean indicating success
        """
        if not error.retry_allowed:
            return False
            
        # Get recovery strategies for this category
        strategies = self._recovery_strategies.get(error.category, [])
        
        # Try each strategy
        for strategy in strategies:
            try:
                if strategy(error):
                    logger.info(f"Successfully recovered from error {error.error_code}")
                    return True
            except Exception as e:
                logger.error(f"Error in recovery strategy: {str(e)}")
                
        return False
        
    def get_recent_errors(self, 
                        count: int = 10,
                        category: ErrorCategory = None,
                        severity: ErrorSeverity = None,
                        component: str = None) -> List[ErrorResponse]:
        """
        Get recent errors, optionally filtered.
        
        Args:
            count: Maximum number of errors to return
            category: Filter by category
            severity: Filter by severity
            component: Filter by component
            
        Returns:
            List of ErrorResponse objects
        """
        with self._lock:
            # Filter errors
            filtered = self._errors
            
            if category:
                filtered = [e for e in filtered if e.category == category]
                
            if severity:
                filtered = [e for e in filtered if e.severity == severity]
                
            if component:
                filtered = [e for e in filtered if e.component == component]
                
            # Return most recent first
            return list(reversed(filtered))[:count]
            
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about errors.
        
        Returns:
            Dictionary with error statistics
        """
        with self._lock:
            errors = self._errors
            
            # Count by category
            categories = {}
            for e in errors:
                if e.category.name not in categories:
                    categories[e.category.name] = 0
                categories[e.category.name] += 1
                
            # Count by severity
            severities = {}
            for e in errors:
                if e.severity.name not in severities:
                    severities[e.severity.name] = 0
                severities[e.severity.name] += 1
                
            # Count by component
            components = {}
            for e in errors:
                if not e.component:
                    continue
                if e.component not in components:
                    components[e.component] = 0
                components[e.component] += 1
                
            return {
                "total_errors": len(errors),
                "by_category": categories,
                "by_severity": severities,
                "by_component": components,
                "has_critical": any(e.severity == ErrorSeverity.CRITICAL for e in errors)
            }
    
    def clear_errors(self) -> None:
        """Clear the error history"""
        with self._lock:
            self._errors = []
            

# Singleton instance
error_manager = ErrorManager()

def handle_exceptions(component: str = None, 
                     retry_count: int = 0,
                     retry_delay: float = 1.0):
    """
    Decorator to handle exceptions in a function.
    
    Args:
        component: Component name for error tracking
        retry_count: Number of retry attempts (0 for no retry)
        retry_delay: Delay between retries in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    
                    # Handle the error
                    error = error_manager.handle_error(e, component)
                    
                    # Check if we should retry
                    if attempts <= retry_count and error.retry_allowed:
                        logger.warning(f"Retry attempt {attempts}/{retry_count} for {func.__name__}")
                        time.sleep(retry_delay)
                    else:
                        # Re-raise the error
                        if isinstance(e, LuminaError):
                            raise e
                        else:
                            # Convert to LuminaError
                            raise LuminaError(
                                message=str(e),
                                category=ErrorCategory.UNKNOWN,
                                component=component,
                                details={"function": func.__name__},
                                exception=e
                            )
        return wrapper
    return decorator 