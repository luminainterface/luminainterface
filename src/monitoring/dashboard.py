"""
Monitoring Dashboard

Simple web dashboard for displaying system metrics and status information.
"""

import os
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.serving import run_simple

from src.monitoring.metrics_system import metrics_manager
from src.error_management.error_handling import error_manager

logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__, 
           template_folder=str(Path(__file__).parent / "templates"),
           static_folder=str(Path(__file__).parent / "static"))

# Dashboard configuration
dashboard_config = {
    "title": "Lumina Monitoring Dashboard",
    "refresh_interval": 10,  # seconds
    "default_time_range": 24  # hours
}

# Define template directory and ensure it exists
template_dir = Path(__file__).parent / "templates"
template_dir.mkdir(parents=True, exist_ok=True)

# Define static directory and ensure it exists
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)

# Create basic template file if it doesn't exist
index_template = template_dir / "index.html"
if not index_template.exists():
    with open(index_template, "w") as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <title>{{ config.title }}</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="{{ url_for('static', filename='dashboard.css') }}">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        const REFRESH_INTERVAL = {{ config.refresh_interval * 1000 }};
        let charts = {};
        
        // Automatic refresh
        setInterval(() => {
            refreshData();
        }, REFRESH_INTERVAL);
        
        // Initial load
        document.addEventListener('DOMContentLoaded', () => {
            refreshData();
            initializeTabs();
        });
        
        function initializeTabs() {
            // Handle tab navigation
            const tabs = document.querySelectorAll('.tab-button');
            const tabContents = document.querySelectorAll('.tab-content');
            
            tabs.forEach(tab => {
                tab.addEventListener('click', () => {
                    const target = tab.getAttribute('data-tab');
                    
                    // Hide all tab contents
                    tabContents.forEach(content => {
                        content.classList.remove('active');
                    });
                    
                    // Deactivate all tabs
                    tabs.forEach(t => {
                        t.classList.remove('active');
                    });
                    
                    // Activate clicked tab and content
                    tab.classList.add('active');
                    document.getElementById(target).classList.add('active');
                });
            });
            
            // Activate first tab by default
            if (tabs.length > 0) {
                tabs[0].click();
            }
        }
        
        async function refreshData() {
            try {
                // Update system overview
                const systemResponse = await fetch('/api/metrics/system');
                const systemData = await systemResponse.json();
                updateSystemOverview(systemData);
                
                // Update performance metrics
                const perfResponse = await fetch('/api/metrics/performance');
                const perfData = await perfResponse.json();
                updatePerformanceMetrics(perfData);
                
                // Update error summary
                const errorResponse = await fetch('/api/errors/summary');
                const errorData = await errorResponse.json();
                updateErrorSummary(errorData);
                
                // Update status indicator
                updateStatusIndicator(systemData, errorData);
                
                // Update last refresh time
                document.getElementById('last-refresh').textContent = 
                    new Date().toLocaleTimeString();
            } catch (error) {
                console.error('Error refreshing data:', error);
            }
        }
        
        function updateSystemOverview(data) {
            document.getElementById('cpu-usage').textContent = 
                data.cpu_usage ? `${data.cpu_usage.toFixed(1)}%` : 'N/A';
            document.getElementById('memory-usage').textContent = 
                data.memory_usage ? `${data.memory_usage.toFixed(1)}%` : 'N/A';
            document.getElementById('disk-usage').textContent = 
                data.disk_usage ? `${data.disk_usage.toFixed(1)}%` : 'N/A';
            
            // Update gauges
            updateGauge('cpu-gauge', data.cpu_usage || 0);
            updateGauge('memory-gauge', data.memory_usage || 0);
            updateGauge('disk-gauge', data.disk_usage || 0);
        }
        
        function updateGauge(id, value) {
            const gauge = document.getElementById(id);
            const percentage = Math.min(100, Math.max(0, value));
            gauge.style.width = `${percentage}%`;
            
            // Update color based on value
            if (percentage < 60) {
                gauge.className = 'gauge-fill gauge-green';
            } else if (percentage < 80) {
                gauge.className = 'gauge-fill gauge-yellow';
            } else {
                gauge.className = 'gauge-fill gauge-red';
            }
        }
        
        function updatePerformanceMetrics(data) {
            // Update or create charts
            createOrUpdateChart('response-time-chart', 
                                'Response Time (ms)', 
                                data.response_time || []);
                                
            createOrUpdateChart('throughput-chart', 
                                'Requests per Minute', 
                                data.throughput || []);
        }
        
        function createOrUpdateChart(id, label, data) {
            const ctx = document.getElementById(id).getContext('2d');
            
            if (charts[id]) {
                // Update existing chart
                charts[id].data.labels = data.map(d => 
                    new Date(d.timestamp).toLocaleTimeString());
                charts[id].data.datasets[0].data = data.map(d => d.value || 0);
                charts[id].update();
            } else {
                // Create new chart
                charts[id] = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: data.map(d => 
                            new Date(d.timestamp).toLocaleTimeString()),
                        datasets: [{
                            label: label,
                            data: data.map(d => d.value || 0),
                            borderColor: '#3498db',
                            backgroundColor: 'rgba(52, 152, 219, 0.1)',
                            tension: 0.4,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                display: true,
                                position: 'top'
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                });
            }
        }
        
        function updateErrorSummary(data) {
            const errorsList = document.getElementById('recent-errors');
            errorsList.innerHTML = '';
            
            if (data.recent_errors && data.recent_errors.length > 0) {
                data.recent_errors.forEach(error => {
                    const li = document.createElement('li');
                    li.className = `error-item error-${error.severity.toLowerCase()}`;
                    li.innerHTML = `
                        <div class="error-code">${error.error_code}</div>
                        <div class="error-message">${error.message}</div>
                        <div class="error-time">
                            ${new Date(error.timestamp).toLocaleTimeString()}
                        </div>
                    `;
                    errorsList.appendChild(li);
                });
            } else {
                const li = document.createElement('li');
                li.textContent = 'No recent errors';
                errorsList.appendChild(li);
            }
            
            // Update error stats
            if (data.statistics) {
                document.getElementById('error-total').textContent = 
                    data.statistics.total_errors || 0;
                    
                // Update error categories chart
                updateErrorCategoriesChart(data.statistics.by_category || {});
            }
        }
        
        function updateErrorCategoriesChart(categories) {
            const ctx = document.getElementById('error-categories-chart').getContext('2d');
            
            // Prepare data
            const labels = Object.keys(categories);
            const values = Object.values(categories);
            
            if (charts['error-categories']) {
                // Update existing chart
                charts['error-categories'].data.labels = labels;
                charts['error-categories'].data.datasets[0].data = values;
                charts['error-categories'].update();
            } else {
                // Create new chart
                charts['error-categories'] = new Chart(ctx, {
                    type: 'doughnut',
                    data: {
                        labels: labels,
                        datasets: [{
                            data: values,
                            backgroundColor: [
                                '#3498db', '#2ecc71', '#e74c3c', '#f1c40f',
                                '#9b59b6', '#1abc9c', '#e67e22', '#34495e'
                            ]
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false
                    }
                });
            }
        }
        
        function updateStatusIndicator(systemData, errorData) {
            const indicator = document.getElementById('system-status');
            
            // Determine status based on system data and errors
            let status = 'green';
            let statusText = 'Healthy';
            
            // Check for critical errors
            if (errorData.statistics && errorData.statistics.has_critical) {
                status = 'red';
                statusText = 'Critical';
            }
            // Check high resource usage
            else if ((systemData.cpu_usage && systemData.cpu_usage > 80) || 
                   (systemData.memory_usage && systemData.memory_usage > 80) ||
                   (systemData.disk_usage && systemData.disk_usage > 80)) {
                status = 'yellow';
                statusText = 'Warning';
            }
            
            // Update indicator
            indicator.className = `status-indicator status-${status}`;
            indicator.textContent = statusText;
        }
    </script>
</head>
<body>
    <header>
        <h1>{{ config.title }}</h1>
        <div class="status-container">
            <div id="system-status" class="status-indicator status-gray">Unknown</div>
            <div class="refresh-info">
                Last refresh: <span id="last-refresh">-</span>
            </div>
        </div>
    </header>
    
    <nav class="tab-navigation">
        <button class="tab-button active" data-tab="overview-tab">Overview</button>
        <button class="tab-button" data-tab="performance-tab">Performance</button>
        <button class="tab-button" data-tab="errors-tab">Errors</button>
    </nav>
    
    <main>
        <section id="overview-tab" class="tab-content active">
            <h2>System Overview</h2>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>CPU Usage</h3>
                    <div class="gauge-container">
                        <div class="gauge-background">
                            <div id="cpu-gauge" class="gauge-fill"></div>
                        </div>
                    </div>
                    <div class="metric-value" id="cpu-usage">-</div>
                </div>
                
                <div class="metric-card">
                    <h3>Memory Usage</h3>
                    <div class="gauge-container">
                        <div class="gauge-background">
                            <div id="memory-gauge" class="gauge-fill"></div>
                        </div>
                    </div>
                    <div class="metric-value" id="memory-usage">-</div>
                </div>
                
                <div class="metric-card">
                    <h3>Disk Usage</h3>
                    <div class="gauge-container">
                        <div class="gauge-background">
                            <div id="disk-gauge" class="gauge-fill"></div>
                        </div>
                    </div>
                    <div class="metric-value" id="disk-usage">-</div>
                </div>
            </div>
        </section>
        
        <section id="performance-tab" class="tab-content">
            <h2>Performance Metrics</h2>
            
            <div class="chart-container">
                <h3>Response Time</h3>
                <div class="chart-wrapper">
                    <canvas id="response-time-chart"></canvas>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>Throughput</h3>
                <div class="chart-wrapper">
                    <canvas id="throughput-chart"></canvas>
                </div>
            </div>
        </section>
        
        <section id="errors-tab" class="tab-content">
            <h2>Error Monitoring</h2>
            
            <div class="error-summary">
                <div class="error-stats">
                    <div class="error-stat-card">
                        <h3>Total Errors</h3>
                        <div class="stat-value" id="error-total">-</div>
                    </div>
                    
                    <div class="error-categories">
                        <h3>Error Categories</h3>
                        <div class="chart-wrapper">
                            <canvas id="error-categories-chart"></canvas>
                        </div>
                    </div>
                </div>
                
                <div class="recent-errors-container">
                    <h3>Recent Errors</h3>
                    <ul id="recent-errors" class="error-list">
                        <li>Loading errors...</li>
                    </ul>
                </div>
            </div>
        </section>
    </main>
    
    <footer>
        <p>Lumina Monitoring System &copy; {{ current_year }}</p>
    </footer>
</body>
</html>""")

# Create CSS file if it doesn't exist
css_file = static_dir / "dashboard.css"
if not css_file.exists():
    with open(css_file, "w") as f:
        f.write("""/* Dashboard styles */
:root {
    --primary-color: #2c3e50;
    --secondary-color: #3498db;
    --background-color: #ecf0f1;
    --card-background: white;
    --text-color: #2c3e50;
    --border-color: #bdc3c7;
    --success-color: #2ecc71;
    --warning-color: #f1c40f;
    --danger-color: #e74c3c;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: var(--background-color);
    color: var(--text-color);
    line-height: 1.6;
}

header {
    background-color: var(--primary-color);
    color: white;
    padding: 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.status-container {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.refresh-info {
    font-size: 0.9rem;
    opacity: 0.8;
}

.status-indicator {
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: bold;
    display: inline-block;
}

.status-green {
    background-color: var(--success-color);
}

.status-yellow {
    background-color: var(--warning-color);
    color: #333;
}

.status-red {
    background-color: var(--danger-color);
}

.status-gray {
    background-color: #95a5a6;
}

nav.tab-navigation {
    background-color: white;
    padding: 0.5rem;
    display: flex;
    border-bottom: 1px solid var(--border-color);
}

.tab-button {
    padding: 0.5rem 1rem;
    border: none;
    background: none;
    cursor: pointer;
    font-size: 1rem;
    border-bottom: 3px solid transparent;
}

.tab-button:hover {
    background-color: rgba(0, 0, 0, 0.05);
}

.tab-button.active {
    border-bottom: 3px solid var(--secondary-color);
    font-weight: bold;
}

main {
    padding: 1rem;
}

.tab-content {
    display: none;
    padding: 1rem;
    background-color: var(--card-background);
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

.tab-content.active {
    display: block;
}

h2 {
    margin-bottom: 1.5rem;
    color: var(--primary-color);
}

h3 {
    margin-bottom: 0.5rem;
    color: var(--primary-color);
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
}

.metric-card {
    background-color: var(--card-background);
    padding: 1rem;
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    text-align: center;
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    margin-top: 0.5rem;
    color: var(--secondary-color);
}

.gauge-container {
    margin: 1rem 0;
}

.gauge-background {
    height: 20px;
    background-color: #ecf0f1;
    border-radius: 10px;
    overflow: hidden;
}

.gauge-fill {
    height: 100%;
    width: 0%;
    transition: width 0.5s ease;
}

.gauge-green {
    background-color: var(--success-color);
}

.gauge-yellow {
    background-color: var(--warning-color);
}

.gauge-red {
    background-color: var(--danger-color);
}

.chart-container {
    margin-bottom: 2rem;
}

.chart-wrapper {
    height: 300px;
    margin-top: 1rem;
}

.error-summary {
    display: grid;
    grid-template-columns: 1fr;
    gap: 1rem;
}

@media (min-width: 768px) {
    .error-summary {
        grid-template-columns: 1fr 1fr;
    }
}

.error-stats {
    display: grid;
    gap: 1rem;
}

.error-stat-card {
    background-color: var(--card-background);
    padding: 1rem;
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    text-align: center;
}

.stat-value {
    font-size: 2rem;
    font-weight: bold;
    color: var(--secondary-color);
}

.error-list {
    list-style: none;
    margin-top: 1rem;
}

.error-item {
    padding: 0.75rem;
    border-left: 4px solid #95a5a6;
    background-color: white;
    margin-bottom: 0.5rem;
    border-radius: 0 3px 3px 0;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.error-critical {
    border-left-color: var(--danger-color);
}

.error-error {
    border-left-color: #e74c3c;
}

.error-warning {
    border-left-color: var(--warning-color);
}

.error-info {
    border-left-color: var(--secondary-color);
}

.error-code {
    font-family: monospace;
    font-size: 0.8rem;
    opacity: 0.7;
}

.error-message {
    font-weight: bold;
}

.error-time {
    font-size: 0.8rem;
    text-align: right;
    opacity: 0.7;
}

footer {
    background-color: var(--primary-color);
    color: white;
    text-align: center;
    padding: 1rem;
    margin-top: 2rem;
}""")


@app.route('/')
def index():
    """Render the dashboard home page"""
    return render_template('index.html', 
                        config=dashboard_config,
                        current_year=datetime.now().year)


@app.route('/api/metrics/system')
def api_system_metrics():
    """Get system metrics"""
    return jsonify(metrics_manager.get_system_metrics())


@app.route('/api/metrics/performance')
def api_performance_metrics():
    """Get performance metrics"""
    hours = int(request.args.get('hours', dashboard_config["default_time_range"]))
    interval = int(request.args.get('interval', 5))  # minutes
    
    # Get aggregated metrics
    response_time = metrics_manager.get_aggregated_history(
        "api", "response_time", interval_minutes=interval, hours=hours)
    
    throughput = metrics_manager.get_aggregated_history(
        "api", "requests", interval_minutes=interval, hours=hours)
    
    # If no data, create sample data for display
    if not response_time:
        # Create sample data for last 24 hours
        now = datetime.now()
        sample_times = [now - timedelta(minutes=5*i) for i in range(12)]
        response_time = [
            {"timestamp": t.isoformat(), "value": 0} 
            for t in reversed(sample_times)
        ]
        
    if not throughput:
        now = datetime.now()
        sample_times = [now - timedelta(minutes=5*i) for i in range(12)]
        throughput = [
            {"timestamp": t.isoformat(), "value": 0} 
            for t in reversed(sample_times)
        ]
    
    return jsonify({
        "response_time": response_time,
        "throughput": throughput
    })


@app.route('/api/errors/summary')
def api_error_summary():
    """Get error summary"""
    # Get recent errors
    count = int(request.args.get('count', 10))
    recent_errors = error_manager.get_recent_errors(count=count)
    
    # If error_manager is not initialized with the singleton pattern, return empty data
    if not recent_errors and not hasattr(error_manager, 'get_error_statistics'):
        return jsonify({
            "recent_errors": [],
            "statistics": {
                "total_errors": 0,
                "by_category": {},
                "by_severity": {},
                "has_critical": False
            }
        })
    
    # Get error statistics
    statistics = error_manager.get_error_statistics()
    
    # Convert error objects to dictionaries
    recent_errors_dict = [
        {
            "error_code": err.error_code,
            "message": err.message,
            "category": err.category.name,
            "severity": err.severity.name,
            "component": err.component,
            "timestamp": err.timestamp
        }
        for err in recent_errors
    ]
    
    return jsonify({
        "recent_errors": recent_errors_dict,
        "statistics": statistics
    })


@app.route('/dream-mode')
def dream_mode_view():
    """Dream Mode visualization dashboard"""
    return render_template('dream_mode.html', config=dashboard_config)


@app.route('/api/metrics/dream-mode')
def api_dream_mode_metrics():
    """API endpoint for Dream Mode metrics"""
    # Get visualization sources
    v7_data = None
    if 'v7' in visualization_sources:
        try:
            v7_data = visualization_sources['v7']()
        except Exception as e:
            logger.error(f"Error getting V7 visualization data: {e}")
    
    # Return dream mode data if available
    if v7_data and "dream_mode" in v7_data:
        return jsonify(v7_data["dream_mode"])
    
    # Default response if not available
    return jsonify({
        "available": False,
        "message": "Dream Mode data not available"
    })


# Visualization data sources
visualization_sources = {}

def register_visualization_source(name: str, callback: Callable) -> None:
    """
    Register a visualization data source
    
    Args:
        name: Unique name for the source
        callback: Function that returns visualization data
    """
    visualization_sources[name] = callback
    logger.info(f"Registered visualization source: {name}")

@app.route('/api/visualization/<source>')
def api_visualization_source(source: str):
    """API endpoint for visualization sources"""
    if source not in visualization_sources:
        return jsonify({"error": f"Visualization source '{source}' not found"}), 404
    
    try:
        data = visualization_sources[source]()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error getting visualization data from source '{source}': {e}")
        return jsonify({"error": f"Error getting visualization data: {str(e)}"}), 500

@app.route('/consciousness')
def consciousness_view():
    """Consciousness visualization dashboard"""
    return render_template('consciousness.html', config=dashboard_config)


@app.route('/language-module')
def language_module_view():
    """Language Module visualization dashboard"""
    return render_template('language_module.html', config=dashboard_config)


@app.route('/api/visualization/language')
def api_language_metrics():
    """API endpoint for Language Module metrics"""
    # Check for language module visualization source
    language_data = None
    
    # First try language-specific source
    if 'language' in visualization_sources:
        try:
            language_data = visualization_sources['language']()
        except Exception as e:
            logger.error(f"Error getting Language module visualization data: {e}")
    
    # Fallback to v7 data if available
    if not language_data and 'v7' in visualization_sources:
        try:
            v7_data = visualization_sources['v7']()
            if v7_data and "language_module" in v7_data:
                language_data = v7_data["language_module"]
        except Exception as e:
            logger.error(f"Error getting V7 visualization data for language module: {e}")
    
    # Return language module data if available
    if language_data:
        return jsonify(language_data)
    
    # Default response if not available
    return jsonify({
        "available": False,
        "message": "Language Module data not available"
    })


@app.route('/api/language/set-weight', methods=['POST'])
def api_set_language_weight():
    """API endpoint to set the language model weight"""
    try:
        data = request.json
        weight = data.get('weight')
        
        if weight is None:
            return jsonify({"error": "Missing weight parameter"}), 400
        
        # Try to find a weight setter in visualization sources
        if 'language_weight_setter' in visualization_sources:
            result = visualization_sources['language_weight_setter'](weight)
            return jsonify({"success": True, "result": result})
        
        # Fallback to v7 handler if available
        elif 'v7_command_handler' in visualization_sources:
            result = visualization_sources['v7_command_handler']('set_language_weight', {'weight': weight})
            return jsonify({"success": True, "result": result})
        
        return jsonify({"error": "No handler available for setting language weight"}), 404
        
    except Exception as e:
        logger.error(f"Error setting language model weight: {e}")
        return jsonify({"error": f"Error setting weight: {str(e)}"}), 500


def start_dashboard(host='127.0.0.1', port=5000, debug=False):
    """Start the monitoring dashboard server"""
    logger.info(f"Starting monitoring dashboard at http://{host}:{port}")
    try:
        run_simple(host, port, app, use_reloader=debug, use_debugger=debug)
    except Exception as e:
        logger.error(f"Error starting dashboard: {str(e)}")


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Start dashboard
    start_dashboard(debug=True) 