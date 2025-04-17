#!/usr/bin/env python3
"""
Memory Synthesis GUI Interface

A graphical interface for interacting with the Memory Synthesis system,
allowing users to visualize and interact with synthesized memories.
"""

import sys
import os
import logging
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memory_synthesis_gui")

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Import required modules
try:
    from src.language_memory_synthesis_integration import LanguageMemorySynthesisIntegration
except ImportError as e:
    logger.error(f"Required module import failed: {str(e)}")
    logger.error("Please ensure all required modules are installed and in the Python path")
    sys.exit(1)

class MemorySynthesisGUI:
    """
    GUI interface for the Memory Synthesis system, allowing users to
    interact with synthesized memories and visualize cross-component connections.
    """
    
    def __init__(self, root):
        """
        Initialize the Memory Synthesis GUI
        
        Args:
            root: The tkinter root window
        """
        self.root = root
        self.root.title("Memory Synthesis Interface")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Set up the integration
        self.integration = None
        self.setup_integration()
        
        # Create the GUI components
        self.create_widgets()
        
        # Store current synthesis results
        self.current_synthesis = None
        
        logger.info("Memory Synthesis GUI initialized")
    
    def setup_integration(self):
        """Set up the memory synthesis integration"""
        try:
            self.integration = LanguageMemorySynthesisIntegration()
            logger.info("Memory synthesis integration initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize memory synthesis integration: {str(e)}")
            messagebox.showerror("Initialization Error", 
                                f"Failed to initialize memory synthesis integration:\n{str(e)}")
    
    def create_widgets(self):
        """Create the GUI widgets"""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create top frame for topic entry and synthesis button
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Topic label and entry
        ttk.Label(top_frame, text="Topic:").pack(side=tk.LEFT, padx=(0, 5))
        self.topic_entry = ttk.Entry(top_frame, width=30)
        self.topic_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.topic_entry.insert(0, "consciousness")
        
        # Synthesize button
        self.synthesize_button = ttk.Button(top_frame, text="Synthesize", command=self.start_synthesis)
        self.synthesize_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Recall button
        self.recall_button = ttk.Button(top_frame, text="Recall Existing", command=self.recall_synthesis)
        self.recall_button.pack(side=tk.LEFT)
        
        # Status label
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        ttk.Label(top_frame, textvariable=self.status_var).pack(side=tk.RIGHT)
        
        # Create a notebook for different views
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Synthesis view
        self.synthesis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.synthesis_frame, text="Synthesis")
        
        # Components view
        self.components_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.components_frame, text="Component Memories")
        
        # Statistics view
        self.stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_frame, text="Statistics")
        
        # Set up synthesis view
        self.setup_synthesis_view()
        
        # Set up components view
        self.setup_components_view()
        
        # Set up statistics view
        self.setup_statistics_view()
        
        # Bottom frame with footer info
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Footer text
        footer_text = "Memory Synthesis Interface - Lumina v10 Neural Network"
        ttk.Label(bottom_frame, text=footer_text).pack(side=tk.LEFT)
        
        # Version info
        version_text = "v0.1.0"
        ttk.Label(bottom_frame, text=version_text).pack(side=tk.RIGHT)
    
    def setup_synthesis_view(self):
        """Set up the synthesis view tab"""
        # Create frame for synthesis info
        info_frame = ttk.LabelFrame(self.synthesis_frame, text="Synthesis Information")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create scrolled text area for synthesis results
        self.synthesis_text = scrolledtext.ScrolledText(info_frame, wrap=tk.WORD, height=20)
        self.synthesis_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.synthesis_text.config(state=tk.DISABLED)
        
        # Create frame for insights
        insights_frame = ttk.LabelFrame(self.synthesis_frame, text="Novel Insights")
        insights_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create insights list
        self.insights_list = tk.Listbox(insights_frame, height=8)
        self.insights_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def setup_components_view(self):
        """Set up the components view tab"""
        # Create frame for components list
        components_list_frame = ttk.Frame(self.components_frame)
        components_list_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Components listbox
        ttk.Label(components_list_frame, text="Memory Components:").pack(anchor=tk.W)
        self.components_list = tk.Listbox(components_list_frame, height=5)
        self.components_list.pack(fill=tk.X, padx=5, pady=5)
        
        # Component memories frame
        component_memories_frame = ttk.LabelFrame(self.components_frame, text="Component Memories")
        component_memories_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Component memories text area
        self.component_memories_text = scrolledtext.ScrolledText(component_memories_frame, wrap=tk.WORD)
        self.component_memories_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.component_memories_text.config(state=tk.DISABLED)
        
        # Set up listbox selection event
        self.components_list.bind('<<ListboxSelect>>', self.on_component_selected)
    
    def setup_statistics_view(self):
        """Set up the statistics view tab"""
        # Create frame for statistics
        stats_main_frame = ttk.Frame(self.stats_frame)
        stats_main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create frame for synthesis stats
        synthesis_stats_frame = ttk.LabelFrame(stats_main_frame, text="Synthesis Statistics")
        synthesis_stats_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create stats text area
        self.stats_text = scrolledtext.ScrolledText(synthesis_stats_frame, wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.stats_text.config(state=tk.DISABLED)
        
        # Create frame for memory stats
        memory_stats_frame = ttk.LabelFrame(stats_main_frame, text="Memory Statistics")
        memory_stats_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Memory stats text
        self.memory_stats_text = scrolledtext.ScrolledText(memory_stats_frame, wrap=tk.WORD, height=8)
        self.memory_stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.memory_stats_text.config(state=tk.DISABLED)
        
        # Refresh button
        self.refresh_stats_button = ttk.Button(stats_main_frame, text="Refresh Statistics", 
                                           command=self.refresh_statistics)
        self.refresh_stats_button.pack(anchor=tk.E, padx=5, pady=5)
    
    def start_synthesis(self):
        """Start the memory synthesis process in a separate thread"""
        topic = self.topic_entry.get().strip()
        if not topic:
            messagebox.showwarning("Empty Topic", "Please enter a topic to synthesize")
            return
        
        # Disable the button and set status
        self.synthesize_button.config(state=tk.DISABLED)
        self.status_var.set(f"Synthesizing topic: {topic}...")
        
        # Clear display areas
        self.clear_display_areas()
        
        # Start synthesis in a thread
        thread = threading.Thread(target=self.perform_synthesis, args=(topic,))
        thread.daemon = True
        thread.start()
    
    def perform_synthesis(self, topic):
        """
        Perform the actual synthesis in a separate thread
        
        Args:
            topic: The topic to synthesize
        """
        try:
            # Perform the synthesis
            results = self.integration.synthesize_topic(topic)
            
            # Store the current synthesis
            self.current_synthesis = results
            
            # Update the UI with results
            self.root.after(0, lambda: self.update_ui_with_synthesis(results))
            
        except Exception as e:
            logger.error(f"Synthesis failed: {str(e)}")
            self.root.after(0, lambda: self.handle_synthesis_error(str(e)))
    
    def update_ui_with_synthesis(self, results):
        """
        Update the UI with synthesis results
        
        Args:
            results: The synthesis results to display
        """
        # Get the synthesized memory
        synthesis = results["synthesis_results"]["synthesized_memory"]
        
        # Update synthesis text
        self.synthesis_text.config(state=tk.NORMAL)
        self.synthesis_text.delete(1.0, tk.END)
        
        # Add core details
        self.synthesis_text.insert(tk.END, f"Topic: {results['topic']}\n\n")
        self.synthesis_text.insert(tk.END, f"Synthesis ID: {synthesis['id']}\n")
        self.synthesis_text.insert(tk.END, f"Created: {synthesis['timestamp']}\n\n")
        
        # Add core understanding
        self.synthesis_text.insert(tk.END, "Core Understanding:\n")
        self.synthesis_text.insert(tk.END, synthesis['core_understanding'] + "\n\n")
        
        # Add integrated components
        self.synthesis_text.insert(tk.END, "Integrated Components:\n")
        for component in synthesis['integrated_components']:
            self.synthesis_text.insert(tk.END, f"- {component}\n")
        
        self.synthesis_text.config(state=tk.DISABLED)
        
        # Update insights list
        self.insights_list.delete(0, tk.END)
        for insight in synthesis["novel_insights"]:
            self.insights_list.insert(tk.END, insight)
        
        # Update components list
        self.components_list.delete(0, tk.END)
        for component in synthesis['integrated_components']:
            self.components_list.insert(tk.END, component)
        
        # Update statistics
        self.refresh_statistics()
        
        # Re-enable the button and update status
        self.synthesize_button.config(state=tk.NORMAL)
        self.status_var.set(f"Synthesis of '{results['topic']}' completed")
    
    def handle_synthesis_error(self, error_message):
        """
        Handle synthesis errors
        
        Args:
            error_message: The error message to display
        """
        messagebox.showerror("Synthesis Error", f"Failed to synthesize memories:\n{error_message}")
        
        # Re-enable the button and update status
        self.synthesize_button.config(state=tk.NORMAL)
        self.status_var.set("Synthesis failed")
    
    def recall_synthesis(self):
        """Recall existing synthesis for the current topic"""
        topic = self.topic_entry.get().strip()
        if not topic:
            messagebox.showwarning("Empty Topic", "Please enter a topic to recall")
            return
        
        try:
            # Retrieve existing synthesis
            synthesis_records = self.integration.get_synthesis_for_topic(topic)
            
            if not synthesis_records:
                messagebox.showinfo("No Records", f"No synthesis records found for topic: {topic}")
                return
            
            # Use the first record
            synthesis = synthesis_records[0]
            
            # Create a mock results object to use with the update function
            results = {
                "topic": topic,
                "synthesis_results": {
                    "synthesized_memory": synthesis
                }
            }
            
            # Store as current synthesis
            self.current_synthesis = results
            
            # Update the UI
            self.update_ui_with_synthesis(results)
            
        except Exception as e:
            logger.error(f"Recall failed: {str(e)}")
            messagebox.showerror("Recall Error", f"Failed to recall synthesis:\n{str(e)}")
    
    def on_component_selected(self, event):
        """
        Handle selection of a component in the list
        
        Args:
            event: The listbox selection event
        """
        if not self.current_synthesis:
            return
            
        # Get the selected component
        selection = self.components_list.curselection()
        if not selection:
            return
            
        component_name = self.components_list.get(selection[0])
        
        # Get the memories for this component
        synthesis = self.current_synthesis["synthesis_results"]
        if "component_memories" not in synthesis:
            return
            
        component_memories = synthesis["component_memories"].get(component_name, [])
        
        # Update the component memories text
        self.component_memories_text.config(state=tk.NORMAL)
        self.component_memories_text.delete(1.0, tk.END)
        
        if not component_memories:
            self.component_memories_text.insert(tk.END, f"No memories found for component: {component_name}")
        else:
            self.component_memories_text.insert(tk.END, f"{len(component_memories)} memories from {component_name}:\n\n")
            
            for i, memory in enumerate(component_memories, 1):
                self.component_memories_text.insert(tk.END, f"{i}. {memory.get('content', 'Unknown content')}\n")
                
                if "metadata" in memory:
                    metadata = memory["metadata"]
                    self.component_memories_text.insert(tk.END, "   Metadata: ")
                    for key, value in metadata.items():
                        self.component_memories_text.insert(tk.END, f"{key}: {value}, ")
                    self.component_memories_text.insert(tk.END, "\n")
                
                self.component_memories_text.insert(tk.END, "\n")
        
        self.component_memories_text.config(state=tk.DISABLED)
    
    def refresh_statistics(self):
        """Refresh the statistics display"""
        if not self.integration:
            return
            
        try:
            # Get statistics
            stats = self.integration.get_stats()
            
            # Update synthesis stats
            synthesis_stats = stats["synthesis_stats"]
            
            self.stats_text.config(state=tk.NORMAL)
            self.stats_text.delete(1.0, tk.END)
            
            self.stats_text.insert(tk.END, f"Total syntheses: {synthesis_stats['synthesis_count']}\n\n")
            
            if synthesis_stats['topics_synthesized']:
                self.stats_text.insert(tk.END, "Topics synthesized:\n")
                for topic in synthesis_stats['topics_synthesized']:
                    self.stats_text.insert(tk.END, f"- {topic}\n")
            
            self.stats_text.config(state=tk.DISABLED)
            
            # Update memory stats
            memory_stats = stats["language_memory_stats"]
            
            self.memory_stats_text.config(state=tk.NORMAL)
            self.memory_stats_text.delete(1.0, tk.END)
            
            self.memory_stats_text.insert(tk.END, f"Language Memory sentences: {memory_stats['sentence_count']}\n")
            
            self.memory_stats_text.config(state=tk.DISABLED)
            
        except Exception as e:
            logger.error(f"Failed to refresh statistics: {str(e)}")
    
    def clear_display_areas(self):
        """Clear all display areas"""
        # Clear synthesis text
        self.synthesis_text.config(state=tk.NORMAL)
        self.synthesis_text.delete(1.0, tk.END)
        self.synthesis_text.config(state=tk.DISABLED)
        
        # Clear insights list
        self.insights_list.delete(0, tk.END)
        
        # Clear components list
        self.components_list.delete(0, tk.END)
        
        # Clear component memories text
        self.component_memories_text.config(state=tk.NORMAL)
        self.component_memories_text.delete(1.0, tk.END)
        self.component_memories_text.config(state=tk.DISABLED)


def main():
    """Main function to start the GUI"""
    root = tk.Tk()
    app = MemorySynthesisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main() 