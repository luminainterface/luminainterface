#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lumina V5 Dashboard Creator
Creates and configures the dashboard for visualizing neural network metrics.
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sqlite3
import threading
import queue
import time
import logging
import argparse
from datetime import datetime
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/dashboard_creator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LuminaDashboardCreator")

class DashboardBuilder:
    """Builds and manages the Lumina V5 Dashboard interface"""
    
    def __init__(self, config_path=None):
        """Initialize the dashboard builder"""
        self.root = None
        self.frames = {}
        self.graphs = {}
        self.data_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.db_path = "data/neural_metrics.db"
        self.config = self.load_config(config_path)
        self.update_interval = self.config.get("update_interval", 1000)  # milliseconds
        
    def load_config(self, config_path):
        """Load dashboard configuration from JSON file"""
        default_config = {
            "title": "Lumina V5 Neural Network Dashboard",
            "width": 1200,
            "height": 800,
            "update_interval": 1000,
            "metrics": [
                {"name": "Neural Activity", "table": "neural_activity"},
                {"name": "Language Processing", "table": "language_metrics"},
                {"name": "Learning Rate", "table": "learning_metrics"},
                {"name": "Memory Usage", "table": "system_metrics"}
            ]
        }
        
        if not config_path or not os.path.exists(config_path):
            logger.info("Using default dashboard configuration")
            return default_config
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return default_config
    
    def setup_database(self):
        """Ensure database exists with required tables"""
        try:
            if not os.path.exists(os.path.dirname(self.db_path)):
                os.makedirs(os.path.dirname(self.db_path))
                
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables if they don't exist
            for metric in self.config["metrics"]:
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {metric['table']} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    value REAL,
                    description TEXT
                )
                """)
            
            conn.commit()
            conn.close()
            logger.info("Database setup complete")
            return True
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            return False
    
    def create_ui(self):
        """Create the main dashboard UI"""
        self.root = tk.Tk()
        self.root.title(self.config["title"])
        self.root.geometry(f"{self.config['width']}x{self.config['height']}")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(header_frame, text=self.config["title"], font=("Arial", 16, "bold")).pack(side=tk.LEFT)
        
        status_frame = ttk.Frame(header_frame)
        status_frame.pack(side=tk.RIGHT)
        
        self.status_label = ttk.Label(status_frame, text="Status: Initializing")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Create tab control
        tab_control = ttk.Notebook(main_container)
        
        # Create tabs
        overview_tab = ttk.Frame(tab_control)
        metrics_tab = ttk.Frame(tab_control)
        settings_tab = ttk.Frame(tab_control)
        
        tab_control.add(overview_tab, text="Overview")
        tab_control.add(metrics_tab, text="Detailed Metrics")
        tab_control.add(settings_tab, text="Settings")
        tab_control.pack(expand=True, fill=tk.BOTH)
        
        # Create overview graphs
        self.create_overview_tab(overview_tab)
        self.create_metrics_tab(metrics_tab)
        self.create_settings_tab(settings_tab)
        
        # Footer
        footer = ttk.Frame(main_container)
        footer.pack(fill=tk.X, pady=5)
        ttk.Label(footer, text="Lumina Neural Network Project © 2023").pack(side=tk.LEFT)
        
        # Set initial status
        self.update_status("Ready")
        
    def create_overview_tab(self, parent):
        """Create the overview dashboard tab"""
        # Create a frame for graphs
        graphs_frame = ttk.Frame(parent)
        graphs_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a 2x2 grid for graphs
        for i, metric in enumerate(self.config["metrics"]):
            frame = ttk.LabelFrame(graphs_frame, text=metric["name"])
            frame.grid(row=i//2, column=i%2, padx=10, pady=10, sticky="nsew")
            
            fig = plt.Figure(figsize=(5, 3), dpi=100)
            ax = fig.add_subplot(111)
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Initial plot
            ax.set_title(metric["name"])
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.grid(True)
            
            # Store reference
            self.graphs[metric["table"]] = {"ax": ax, "canvas": canvas, "data": []}
        
        # Configure grid weights
        for i in range(2):
            graphs_frame.columnconfigure(i, weight=1)
            graphs_frame.rowconfigure(i, weight=1)
    
    def create_metrics_tab(self, parent):
        """Create the detailed metrics tab"""
        # Create a frame for detailed metrics
        details_frame = ttk.Frame(parent)
        details_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a notebook for each metric
        metrics_notebook = ttk.Notebook(details_frame)
        metrics_notebook.pack(fill=tk.BOTH, expand=True)
        
        for metric in self.config["metrics"]:
            tab = ttk.Frame(metrics_notebook)
            metrics_notebook.add(tab, text=metric["name"])
            
            # Add a Treeview to display data
            columns = ("timestamp", "value", "description")
            tree = ttk.Treeview(tab, columns=columns, show="headings")
            
            tree.heading("timestamp", text="Timestamp")
            tree.heading("value", text="Value")
            tree.heading("description", text="Description")
            
            tree.column("timestamp", width=200)
            tree.column("value", width=100)
            tree.column("description", width=300)
            
            scrollbar = ttk.Scrollbar(tab, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Store reference
            self.frames[metric["table"]] = tree
    
    def create_settings_tab(self, parent):
        """Create the settings tab"""
        settings_frame = ttk.Frame(parent)
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Update interval setting
        interval_frame = ttk.Frame(settings_frame)
        interval_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(interval_frame, text="Update Interval (ms):").pack(side=tk.LEFT)
        
        interval_var = tk.StringVar(value=str(self.update_interval))
        interval_entry = ttk.Entry(interval_frame, textvariable=interval_var, width=10)
        interval_entry.pack(side=tk.LEFT, padx=10)
        
        def update_interval():
            try:
                self.update_interval = int(interval_var.get())
                messagebox.showinfo("Settings Updated", "Update interval changed successfully")
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number")
        
        ttk.Button(interval_frame, text="Apply", command=update_interval).pack(side=tk.LEFT)
        
        # Database path setting
        db_frame = ttk.Frame(settings_frame)
        db_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(db_frame, text="Database Path:").pack(side=tk.LEFT)
        
        db_var = tk.StringVar(value=self.db_path)
        db_entry = ttk.Entry(db_frame, textvariable=db_var, width=40)
        db_entry.pack(side=tk.LEFT, padx=10)
        
        def update_db_path():
            self.db_path = db_var.get()
            if self.setup_database():
                messagebox.showinfo("Settings Updated", "Database path updated successfully")
            else:
                messagebox.showerror("Error", "Failed to connect to database")
        
        ttk.Button(db_frame, text="Apply", command=update_db_path).pack(side=tk.LEFT)
        
        # Reset button
        reset_frame = ttk.Frame(settings_frame)
        reset_frame.pack(fill=tk.X, pady=20)
        
        def reset_dashboard():
            if messagebox.askyesno("Reset Dashboard", "Are you sure you want to reset all dashboard settings?"):
                self.update_interval = 1000
                interval_var.set("1000")
                messagebox.showinfo("Reset Complete", "Dashboard settings have been reset")
        
        ttk.Button(reset_frame, text="Reset Dashboard Settings", command=reset_dashboard).pack()
    
    def update_status(self, message):
        """Update the status message in the UI"""
        self.status_label.config(text=f"Status: {message}")
        logger.info(f"Dashboard status: {message}")
    
    def fetch_data(self):
        """Background thread to fetch data from database"""
        while not self.stop_event.is_set():
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                for metric in self.config["metrics"]:
                    table = metric["table"]
                    cursor.execute(f"""
                    SELECT timestamp, value, description 
                    FROM {table} 
                    ORDER BY timestamp DESC 
                    LIMIT 100
                    """)
                    data = cursor.fetchall()
                    self.data_queue.put({"table": table, "data": data})
                
                conn.close()
                time.sleep(self.update_interval / 1000)  # Convert ms to seconds
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                time.sleep(5)  # Wait before retry
    
    def update_ui(self):
        """Update UI with new data from the queue"""
        try:
            while not self.data_queue.empty():
                item = self.data_queue.get_nowait()
                table = item["table"]
                data = item["data"]
                
                # Update graph
                if table in self.graphs:
                    graph = self.graphs[table]
                    ax = graph["ax"]
                    
                    # Clear previous data
                    ax.clear()
                    
                    if data:
                        # Prepare data for plotting (convert to proper format)
                        timestamps = [datetime.fromisoformat(row[0]) for row in data]
                        values = [row[1] for row in data]
                        
                        # Plot new data
                        ax.plot(timestamps, values, marker='o')
                        ax.set_title(f"{table.replace('_', ' ').title()}")
                        ax.set_xlabel("Time")
                        ax.set_ylabel("Value")
                        ax.grid(True)
                        
                        # Format x-axis with dates
                        fig = ax.figure
                        fig.autofmt_xdate()
                        
                        # Redraw canvas
                        graph["canvas"].draw()
                
                # Update treeview
                if table in self.frames:
                    tree = self.frames[table]
                    
                    # Clear previous data
                    for item in tree.get_children():
                        tree.delete(item)
                    
                    # Add new data
                    for row in data:
                        tree.insert("", "end", values=row)
            
            # Schedule next update
            if not self.stop_event.is_set():
                self.root.after(100, self.update_ui)
        except Exception as e:
            logger.error(f"Error updating UI: {e}")
            if not self.stop_event.is_set():
                self.root.after(1000, self.update_ui)
    
    def on_closing(self):
        """Handle window closing event"""
        if messagebox.askokcancel("Quit", "Do you want to quit the dashboard?"):
            self.stop_event.set()
            self.root.destroy()
    
    def run(self):
        """Run the dashboard application"""
        if not self.setup_database():
            logger.error("Failed to setup database, exiting")
            return False
        
        self.create_ui()
        
        # Start background thread for data fetching
        data_thread = threading.Thread(target=self.fetch_data, daemon=True)
        data_thread.start()
        
        # Schedule UI updates
        self.root.after(100, self.update_ui)
        
        # Start main loop
        self.root.mainloop()
        
        return True

def main():
    """Main entry point for the dashboard creator"""
    parser = argparse.ArgumentParser(description="Lumina V5 Dashboard Creator")
    parser.add_argument("--config", help="Path to dashboard configuration file")
    parser.add_argument("--db", help="Path to neural metrics database")
    args = parser.parse_args()
    
    logger.info("Starting Lumina V5 Dashboard Creator")
    
    dashboard = DashboardBuilder(config_path=args.config)
    if args.db:
        dashboard.db_path = args.db
    
    result = dashboard.run()
    
    if result:
        logger.info("Dashboard exited successfully")
        return 0
    else:
        logger.error("Dashboard failed to start properly")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 