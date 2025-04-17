import os
import psutil
import platform
import GPUtil
from typing import Dict, Any
from datetime import datetime

def get_system_info() -> Dict[str, Any]:
    """Gather system information for dynamic configuration"""
    system_info = {
        'hardware': {
            'cpu': {
                'cores': psutil.cpu_count(logical=False),
                'threads': psutil.cpu_count(logical=True),
                'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else None,
                'usage': psutil.cpu_percent(interval=1)
            },
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'used_percent': psutil.virtual_memory().percent
            },
            'gpu': get_gpu_info(),
            'os': {
                'name': platform.system(),
                'version': platform.version(),
                'architecture': platform.architecture()[0]
            },
            'display': get_display_info()
        },
        'growth': {
            'start_time': datetime.now().isoformat(),
            'uptime': psutil.boot_time(),
            'process_count': len(psutil.Process().children()),
            'system_load': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
        },
        'evolution': {
            'stages_completed': 0,
            'current_stage': 'SEED',
            'growth_rate': calculate_growth_rate(),
            'stability': calculate_system_stability()
        }
    }
    return system_info

def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information if available"""
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            return {
                'count': len(gpus),
                'memory_total': gpus[0].memoryTotal,
                'memory_free': gpus[0].memoryFree,
                'name': gpus[0].name,
                'load': gpus[0].load * 100 if gpus[0].load else 0
            }
    except:
        pass
    return {'count': 0}

def get_display_info() -> Dict[str, Any]:
    """Get display information"""
    try:
        from screeninfo import get_monitors
        monitors = get_monitors()
        if monitors:
            return {
                'count': len(monitors),
                'primary_resolution': {
                    'width': monitors[0].width,
                    'height': monitors[0].height
                },
                'refresh_rate': monitors[0].frequency if hasattr(monitors[0], 'frequency') else 60
            }
    except:
        pass
    return {'count': 1}

def calculate_growth_rate() -> float:
    """Calculate system growth rate based on various factors"""
    try:
        # Consider CPU usage, memory usage, and process count
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        process_count = len(psutil.Process().children())
        
        # Normalize factors
        cpu_factor = 1.0 - (cpu_usage / 100)
        memory_factor = 1.0 - (memory_usage / 100)
        process_factor = min(1.0, process_count / 100)
        
        # Calculate growth rate (0.0 to 1.0)
        return (cpu_factor + memory_factor + process_factor) / 3
    except:
        return 0.5  # Default growth rate

def calculate_system_stability() -> float:
    """Calculate system stability based on resource usage patterns"""
    try:
        # Get CPU usage history
        cpu_history = [psutil.cpu_percent(interval=0.1) for _ in range(5)]
        
        # Calculate variance in CPU usage
        cpu_variance = sum((x - sum(cpu_history)/len(cpu_history))**2 for x in cpu_history) / len(cpu_history)
        
        # Get memory usage
        memory_usage = psutil.virtual_memory().percent
        
        # Calculate stability (0.0 to 1.0)
        cpu_stability = 1.0 - min(1.0, cpu_variance / 100)
        memory_stability = 1.0 - (memory_usage / 100)
        
        return (cpu_stability + memory_stability) / 2
    except:
        return 0.5  # Default stability

def calculate_optimal_settings(system_info: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate optimal settings based on system capabilities"""
    hardware = system_info['hardware']
    growth = system_info['growth']
    evolution = system_info['evolution']
    
    # Calculate factors based on hardware
    cpu_factor = min(1.0, hardware['cpu']['cores'] / 4)
    memory_factor = min(1.0, hardware['memory']['available'] / (8 * 1024 * 1024 * 1024))
    gpu_factor = 1.0 if hardware['gpu']['count'] > 0 else 0.5
    display_factor = min(1.0, hardware['display']['primary_resolution']['width'] / 1920)
    
    # Calculate growth-based factors
    growth_factor = evolution['growth_rate']
    stability_factor = evolution['stability']
    
    return {
        'network': {
            'num_layers': max(2, min(6, int(3 * cpu_factor * growth_factor))),
            'nodes_per_layer': max(3, min(8, int(4 * memory_factor * growth_factor))),
            'animation_speed': 1.0 * gpu_factor * stability_factor,
            'signal_frequency': 0.5 * gpu_factor * growth_factor,
            'complexity': 0.5 * (cpu_factor + memory_factor) / 2 * growth_factor
        },
        'appearance': {
            'node_size': max(20, min(40, int(30 * display_factor))),
            'connection_width': max(1, min(3, int(2 * display_factor))),
            'background_color': '#1E1E1E',
            'grid_enabled': True,
            'node_colors': {
                'normal': '#3498db',
                'auto_learner': '#2ecc71',
                'logic_gate': '#e74c3c',
                'seed': '#f39c12',
                'system': '#9b59b6'  # New color for system nodes
            },
            'connection_colors': {
                'literal': '#3498db',
                'semantic': '#e74c3c',
                'hybrid': '#9b59b6',
                'auto_learner': '#2ecc71',
                'logic_gate': '#e67e22',
                'seed': '#f1c40f',
                'system': '#8e44ad'  # New color for system connections
            }
        },
        'data_sources': {
            'autowiki': {
                'enabled': False,
                'update_interval': max(500, min(2000, int(1000 / (cpu_factor * stability_factor))))
            },
            'neural_seed': {
                'enabled': False,
                'update_interval': max(500, min(2000, int(1000 / (cpu_factor * stability_factor))))
            },
            'external': {
                'enabled': False,
                'update_interval': max(500, min(2000, int(1000 / (cpu_factor * stability_factor))))
            },
            'system': {  # New data source for system growth
                'enabled': True,
                'update_interval': max(250, min(1000, int(500 / stability_factor)))
            }
        },
        'animation': {
            'base_frequency': 0.5 * gpu_factor * growth_factor,
            'frequency_variance': 0.2 * (1.0 - stability_factor),
            'movement_speed': 1.0 * gpu_factor * growth_factor,
            'movement_radius': max(5.0, min(15.0, 10.0 * display_factor * growth_factor)),
            'transition_speed': 0.1 * gpu_factor * stability_factor
        },
        'growth': {  # New growth settings
            'rate': growth_factor,
            'stability': stability_factor,
            'stage': evolution['current_stage'],
            'metrics_update_interval': max(100, min(1000, int(500 / stability_factor)))
        }
    }

def generate_config() -> Dict[str, Any]:
    """Generate configuration based on system capabilities"""
    system_info = get_system_info()
    return calculate_optimal_settings(system_info) 