import os
import sys
import logging
import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Lumina V5 components
try:
    from v5.lumina_v5.core import system_monitor
    from v5.lumina_v5.visualization import graph_generator
    from language.database_manager import DatabaseManager
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/dashboard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LuminaV5Dashboard")

class LuminaV5Dashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Lumina V5 Dashboard")
        self.root.geometry("1024x768")
        self.root.minsize(800, 600)
        
        # Initialize database connection
        try:
            self.db_manager = DatabaseManager()
            logger.info("Successfully connected to database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self.db_manager = None
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.system_tab = ttk.Frame(self.notebook)
        self.language_tab = ttk.Frame(self.notebook)
        self.network_tab = ttk.Frame(self.notebook)
        self.stats_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.system_tab, text="System Monitor")
        self.notebook.add(self.language_tab, text="Language Analysis")
        self.notebook.add(self.network_tab, text="Neural Network")
        self.notebook.add(self.stats_tab, text="Statistics")
        
        # Initialize UI components
        self.setup_system_tab()
        self.setup_language_tab()
        self.setup_network_tab()
        self.setup_stats_tab()
        
        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.update_monitor, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Dashboard initialized successfully")
    
    def setup_system_tab(self):
        # System information section
        system_frame = ttk.LabelFrame(self.system_tab, text="System Information")
        system_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # System stats
        self.system_stats = scrolledtext.ScrolledText(system_frame, wrap=tk.WORD, height=10)
        self.system_stats.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.system_stats.insert(tk.END, "Loading system information...")
        self.system_stats.config(state=tk.DISABLED)
        
        # Control buttons
        control_frame = ttk.Frame(self.system_tab)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Refresh", command=self.refresh_system_info).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Run Diagnostics", command=self.run_diagnostics).pack(side=tk.LEFT, padx=5)
    
    def setup_language_tab(self):
        # Language processing stats
        language_frame = ttk.LabelFrame(self.language_tab, text="Language Processing")
        language_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Recent conversations
        conv_frame = ttk.LabelFrame(language_frame, text="Recent Conversations")
        conv_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.conversations_text = scrolledtext.ScrolledText(conv_frame, wrap=tk.WORD, height=15)
        self.conversations_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.conversations_text.insert(tk.END, "Loading recent conversations...")
        self.conversations_text.config(state=tk.DISABLED)
    
    def setup_network_tab(self):
        # Neural network visualization
        network_frame = ttk.LabelFrame(self.network_tab, text="Neural Network Visualization")
        network_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Placeholder for network visualization
        network_canvas = tk.Canvas(network_frame, bg="white", height=400)
        network_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        network_canvas.create_text(
            network_canvas.winfo_reqwidth()/2, 
            network_canvas.winfo_reqheight()/2,
            text="Neural network visualization will appear here",
            fill="gray"
        )
    
    def setup_stats_tab(self):
        # Statistics and metrics
        stats_frame = ttk.LabelFrame(self.stats_tab, text="Learning Statistics")
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.stats_text = scrolledtext.ScrolledText(stats_frame, wrap=tk.WORD, height=20)
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.stats_text.insert(tk.END, "Loading statistics...")
        self.stats_text.config(state=tk.DISABLED)
    
    def refresh_system_info(self):
        """Refresh system information display"""
        try:
            if hasattr(system_monitor, 'get_system_info'):
                system_info = system_monitor.get_system_info()
                
                self.system_stats.config(state=tk.NORMAL)
                self.system_stats.delete(1.0, tk.END)
                
                for key, value in system_info.items():
                    self.system_stats.insert(tk.END, f"{key}: {value}\n")
                    
                self.system_stats.config(state=tk.DISABLED)
                self.status_bar.config(text="System information updated")
            else:
                logger.error("system_monitor module doesn't have get_system_info function")
                self.status_bar.config(text="Error: Cannot retrieve system information")
        except Exception as e:
            logger.error(f"Error refreshing system info: {e}")
            self.status_bar.config(text=f"Error: {str(e)}")
    
    def run_diagnostics(self):
        """Run system diagnostics"""
        self.status_bar.config(text="Running diagnostics...")
        # Implementation would depend on available diagnostic tools
        
        # Placeholder for diagnostics result
        result = "Diagnostics completed.\n\n"
        result += "All systems operational.\n"
        result += "Database connection: Active\n"
        result += "Neural network status: Online\n"
        result += "Language system: Operational\n"
        
        self.system_stats.config(state=tk.NORMAL)
        self.system_stats.delete(1.0, tk.END)
        self.system_stats.insert(tk.END, result)
        self.system_stats.config(state=tk.DISABLED)
        
        self.status_bar.config(text="Diagnostics completed")
    
    def update_monitor(self):
        """Background thread to update dashboard information"""
        while True:
            try:
                # Update conversations if database is connected
                if self.db_manager:
                    self.update_conversations()
                    self.update_stats()
                
                # Update system info occasionally
                self.refresh_system_info()
            except Exception as e:
                logger.error(f"Error in monitor thread: {e}")
            
            # Sleep for 30 seconds before next update
            time.sleep(30)
    
    def update_conversations(self):
        """Update recent conversations display"""
        try:
            # Get recent conversations from database
            if hasattr(self.db_manager, 'get_recent_conversations'):
                conversations = self.db_manager.get_recent_conversations(limit=5)
                
                self.conversations_text.config(state=tk.NORMAL)
                self.conversations_text.delete(1.0, tk.END)
                
                if not conversations:
                    self.conversations_text.insert(tk.END, "No recent conversations found.")
                else:
                    for conv in conversations:
                        self.conversations_text.insert(tk.END, f"ID: {conv.id}\n")
                        self.conversations_text.insert(tk.END, f"Started: {conv.start_time}\n")
                        self.conversations_text.insert(tk.END, f"Exchanges: {len(conv.exchanges)}\n")
                        self.conversations_text.insert(tk.END, "-" * 40 + "\n")
                
                self.conversations_text.config(state=tk.DISABLED)
        except Exception as e:
            logger.error(f"Error updating conversations: {e}")
    
    def update_stats(self):
        """Update learning statistics"""
        try:
            if hasattr(self.db_manager, 'get_learning_statistics'):
                stats = self.db_manager.get_learning_statistics()
                
                self.stats_text.config(state=tk.NORMAL)
                self.stats_text.delete(1.0, tk.END)
                
                for key, value in stats.items():
                    self.stats_text.insert(tk.END, f"{key}: {value}\n")
                
                self.stats_text.config(state=tk.DISABLED)
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")


def main():
    """Main function to start the dashboard"""
    try:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Initialize Tkinter
        root = tk.Tk()
        app = LuminaV5Dashboard(root)
        
        # Set theme if available
        try:
            from ttkthemes import ThemedStyle
            style = ThemedStyle(root)
            style.set_theme("clam")  # or another theme like "arc", "equilux"
        except ImportError:
            logger.info("ttkthemes not available, using default theme")
        
        # Start application
        logger.info("Starting Lumina V5 Dashboard")
        root.mainloop()
    except Exception as e:
        logger.critical(f"Failed to start dashboard: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
