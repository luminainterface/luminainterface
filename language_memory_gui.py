#!/usr/bin/env python3
"""
Language Memory GUI

A simple GUI for interacting with the language memory system.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from pathlib import Path
import threading
import logging
import json

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("language_memory_gui")

# Try to import required components
try:
    from src.language_memory_synthesis_integration import LanguageMemorySynthesisIntegration
except ImportError as e:
    logger.error(f"Failed to import required component: {str(e)}")
    logger.error("Please ensure the language_memory_synthesis_integration.py file exists")
    sys.exit(1)


class LanguageMemoryGUI:
    """
    GUI for the Language Memory System
    
    Provides a simple interface for:
    - Storing new memories
    - Searching existing memories
    - Synthesizing topics across memory components
    - Viewing memory statistics
    """
    
    def __init__(self, root):
        """
        Initialize the Language Memory GUI
        
        Args:
            root: The tkinter root window
        """
        self.root = root
        self.root.title("Language Memory System")
        self.root.geometry("900x700")
        
        # Initialize memory system
        self.initialize_memory_system()
        
        # Create the main frame
        self.create_gui_elements()
        
        logger.info("Language Memory GUI initialized")
    
    def initialize_memory_system(self):
        """Initialize the memory system integration"""
        try:
            self.memory_system = LanguageMemorySynthesisIntegration()
            logger.info("Memory system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize memory system: {str(e)}")
            messagebox.showerror("Initialization Error", 
                               f"Failed to initialize memory system: {str(e)}\n\n"
                               f"The application may have limited functionality.")
    
    def create_gui_elements(self):
        """Create all GUI elements"""
        # Create a notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_memory_tab()
        self.create_synthesis_tab()
        self.create_stats_tab()
    
    def create_memory_tab(self):
        """Create the memory storage and retrieval tab"""
        memory_frame = ttk.Frame(self.notebook)
        self.notebook.add(memory_frame, text="Memory")
        
        # Left side - Input new memories
        input_frame = ttk.LabelFrame(memory_frame, text="Store New Memory")
        input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Memory input area
        ttk.Label(input_frame, text="Enter Memory Content:").pack(pady=(10, 0))
        self.memory_text = scrolledtext.ScrolledText(input_frame, height=10)
        self.memory_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Topic and emotion inputs
        ttk.Label(input_frame, text="Topic:").pack(pady=(5, 0))
        self.topic_entry = ttk.Entry(input_frame)
        self.topic_entry.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(input_frame, text="Emotion:").pack(pady=(5, 0))
        self.emotion_entry = ttk.Entry(input_frame)
        self.emotion_entry.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(input_frame, text="Keywords (comma separated):").pack(pady=(5, 0))
        self.keywords_entry = ttk.Entry(input_frame)
        self.keywords_entry.pack(fill=tk.X, padx=5, pady=2)
        
        # Store button
        self.store_button = ttk.Button(input_frame, text="Store Memory", command=self.store_memory)
        self.store_button.pack(pady=10)
        
        # Right side - Search memories
        search_frame = ttk.LabelFrame(memory_frame, text="Search Memories")
        search_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Search options
        search_options_frame = ttk.Frame(search_frame)
        search_options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(search_options_frame, text="Search By:").pack(side=tk.LEFT)
        
        self.search_type = tk.StringVar(value="topic")
        search_types = ["topic", "keyword", "text"]
        
        for i, search_type in enumerate(search_types):
            ttk.Radiobutton(search_options_frame, text=search_type.capitalize(), 
                          variable=self.search_type, value=search_type).pack(side=tk.LEFT, padx=10)
        
        ttk.Label(search_frame, text="Search Query:").pack(pady=(5, 0))
        self.search_entry = ttk.Entry(search_frame)
        self.search_entry.pack(fill=tk.X, padx=5, pady=2)
        
        # Search button
        self.search_button = ttk.Button(search_frame, text="Search", command=self.search_memories)
        self.search_button.pack(pady=5)
        
        # Results area
        ttk.Label(search_frame, text="Search Results:").pack(pady=(5, 0))
        self.results_text = scrolledtext.ScrolledText(search_frame, height=10)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def create_synthesis_tab(self):
        """Create the memory synthesis tab"""
        synthesis_frame = ttk.Frame(self.notebook)
        self.notebook.add(synthesis_frame, text="Synthesis")
        
        # Topic input
        input_frame = ttk.Frame(synthesis_frame)
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(input_frame, text="Topic to Synthesize:").pack(side=tk.LEFT, padx=5)
        self.synthesis_topic_entry = ttk.Entry(input_frame, width=30)
        self.synthesis_topic_entry.pack(side=tk.LEFT, padx=5)
        
        # Depth selector
        ttk.Label(input_frame, text="Depth:").pack(side=tk.LEFT, padx=5)
        self.depth_var = tk.IntVar(value=3)
        depth_spinner = ttk.Spinbox(input_frame, from_=1, to=5, width=5, textvariable=self.depth_var)
        depth_spinner.pack(side=tk.LEFT, padx=5)
        
        # Synthesize button
        self.synthesize_button = ttk.Button(input_frame, text="Synthesize", 
                                          command=self.synthesize_topic)
        self.synthesize_button.pack(side=tk.LEFT, padx=15)
        
        # Results area
        results_frame = ttk.LabelFrame(synthesis_frame, text="Synthesis Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.synthesis_results = scrolledtext.ScrolledText(results_frame)
        self.synthesis_results.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def create_stats_tab(self):
        """Create the statistics tab"""
        stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(stats_frame, text="Statistics")
        
        # Controls
        controls_frame = ttk.Frame(stats_frame)
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.refresh_stats_button = ttk.Button(controls_frame, text="Refresh Statistics", 
                                             command=self.refresh_stats)
        self.refresh_stats_button.pack(side=tk.LEFT)
        
        # Stats display area
        stats_display_frame = ttk.Frame(stats_frame)
        stats_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create a treeview for statistics
        self.stats_tree = ttk.Treeview(stats_display_frame)
        self.stats_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(stats_display_frame, orient="vertical", 
                                command=self.stats_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.stats_tree.configure(yscrollcommand=scrollbar.set)
        
        # Configure the treeview
        self.stats_tree["columns"] = ("value")
        self.stats_tree.column("#0", width=250, minwidth=150)
        self.stats_tree.column("value", width=200, minwidth=150)
        
        self.stats_tree.heading("#0", text="Statistic")
        self.stats_tree.heading("value", text="Value")
        
        # Load initial stats
        self.refresh_stats()
    
    def store_memory(self):
        """Store a new memory from user input"""
        # Get input values
        memory_content = self.memory_text.get("1.0", tk.END).strip()
        topic = self.topic_entry.get().strip()
        emotion = self.emotion_entry.get().strip()
        keywords = [k.strip() for k in self.keywords_entry.get().split(",") if k.strip()]
        
        if not memory_content:
            messagebox.showwarning("Input Required", "Please enter memory content")
            return
        
        try:
            # Check if we have a conversation memory component
            if "conversation_memory" in self.memory_system.components:
                memory_component = self.memory_system.components["conversation_memory"]
                
                # Store the memory
                result = memory_component.store(
                    content=memory_content,
                    metadata={
                        "topic": topic,
                        "emotion": emotion,
                        "keywords": keywords,
                        "source": "gui_input"
                    }
                )
                
                # Clear input fields
                self.memory_text.delete("1.0", tk.END)
                self.topic_entry.delete(0, tk.END)
                self.emotion_entry.delete(0, tk.END)
                self.keywords_entry.delete(0, tk.END)
                
                messagebox.showinfo("Success", "Memory stored successfully")
                logger.info(f"Memory stored with ID: {result.get('id', 'unknown')}")
            else:
                messagebox.showwarning("Component Missing", 
                                      "Conversation memory component is not available")
        except Exception as e:
            error_msg = f"Error storing memory: {str(e)}"
            logger.error(error_msg)
            messagebox.showerror("Error", error_msg)
    
    def search_memories(self):
        """Search for memories based on user input"""
        search_query = self.search_entry.get().strip()
        search_type = self.search_type.get()
        
        if not search_query:
            messagebox.showwarning("Input Required", "Please enter a search query")
            return
        
        try:
            # Check if we have a conversation memory component
            if "conversation_memory" in self.memory_system.components:
                memory_component = self.memory_system.components["conversation_memory"]
                
                # Perform search based on selected type
                if search_type == "topic":
                    results = memory_component.retrieve_by_topic(search_query)
                elif search_type == "keyword":
                    results = memory_component.retrieve_by_keyword(search_query)
                elif search_type == "text":
                    results = memory_component.search_text(search_query)
                else:
                    results = []
                
                # Display results
                self.results_text.delete("1.0", tk.END)
                
                if results:
                    self.results_text.insert(tk.END, f"Found {len(results)} memories:\n\n")
                    
                    for i, memory in enumerate(results):
                        self.results_text.insert(tk.END, f"Memory {i+1}:\n")
                        self.results_text.insert(tk.END, f"Content: {memory.get('content', 'No content')}\n")
                        
                        if "metadata" in memory:
                            metadata = memory["metadata"]
                            self.results_text.insert(tk.END, f"Topic: {metadata.get('topic', 'N/A')}\n")
                            self.results_text.insert(tk.END, f"Emotion: {metadata.get('emotion', 'N/A')}\n")
                            
                            if "keywords" in metadata and metadata["keywords"]:
                                self.results_text.insert(tk.END, f"Keywords: {', '.join(metadata['keywords'])}\n")
                        
                        self.results_text.insert(tk.END, f"Timestamp: {memory.get('timestamp', 'N/A')}\n\n")
                else:
                    self.results_text.insert(tk.END, "No results found.")
                
                logger.info(f"Search completed for '{search_query}' with {len(results)} results")
            else:
                messagebox.showwarning("Component Missing", 
                                      "Conversation memory component is not available")
        except Exception as e:
            error_msg = f"Error searching memories: {str(e)}"
            logger.error(error_msg)
            messagebox.showerror("Error", error_msg)
    
    def synthesize_topic(self):
        """Synthesize memories around a topic"""
        topic = self.synthesis_topic_entry.get().strip()
        depth = self.depth_var.get()
        
        if not topic:
            messagebox.showwarning("Input Required", "Please enter a topic to synthesize")
            return
        
        # Disable button while processing
        self.synthesize_button.configure(state="disabled")
        self.synthesis_results.delete("1.0", tk.END)
        self.synthesis_results.insert(tk.END, f"Synthesizing topic '{topic}'... Please wait.\n")
        
        # Run synthesis in a separate thread
        threading.Thread(target=self._run_synthesis, args=(topic, depth)).start()
    
    def _run_synthesis(self, topic, depth):
        """
        Run the synthesis operation in a background thread
        
        Args:
            topic: Topic to synthesize
            depth: Search depth
        """
        try:
            # Perform the synthesis
            results = self.memory_system.synthesize_topic(topic, depth)
            
            # Schedule the results display on the main thread
            self.root.after(0, lambda: self._display_synthesis_results(results))
        except Exception as e:
            error_msg = f"Error during synthesis: {str(e)}"
            logger.error(error_msg)
            self.root.after(0, lambda: messagebox.showerror("Synthesis Error", error_msg))
            self.root.after(0, lambda: self.synthesize_button.configure(state="normal"))
    
    def _display_synthesis_results(self, results):
        """
        Display the synthesis results in the GUI
        
        Args:
            results: The results from the synthesis operation
        """
        self.synthesis_results.delete("1.0", tk.END)
        
        if "synthesis_results" in results and results["synthesis_results"]:
            synthesis = results["synthesis_results"]["synthesized_memory"]
            
            self.synthesis_results.insert(tk.END, "SYNTHESIS RESULTS\n")
            self.synthesis_results.insert(tk.END, "================\n\n")
            
            self.synthesis_results.insert(tk.END, f"Topic: {', '.join(synthesis['topics'])}\n")
            self.synthesis_results.insert(tk.END, f"ID: {synthesis['id']}\n")
            self.synthesis_results.insert(tk.END, f"Created: {synthesis['timestamp']}\n\n")
            
            self.synthesis_results.insert(tk.END, "CORE UNDERSTANDING:\n")
            self.synthesis_results.insert(tk.END, f"{synthesis['core_understanding']}\n\n")
            
            self.synthesis_results.insert(tk.END, "NOVEL INSIGHTS:\n")
            for insight in synthesis['novel_insights']:
                self.synthesis_results.insert(tk.END, f"• {insight}\n")
            
            self.synthesis_results.insert(tk.END, "\nCOMPONENT CONTRIBUTIONS:\n")
            for component, count in synthesis['component_contributions'].items():
                self.synthesis_results.insert(tk.END, f"• {component}: {count} items\n")
            
            # Display related topics if available
            if "related_topics" in results["synthesis_results"]:
                related = results["synthesis_results"]["related_topics"]
                self.synthesis_results.insert(tk.END, f"\nRELATED TOPICS: {', '.join(related)}\n")
        
        elif "errors" in results and results["errors"]:
            self.synthesis_results.insert(tk.END, "ERROR DURING SYNTHESIS:\n\n")
            for error in results["errors"]:
                self.synthesis_results.insert(tk.END, f"• {error}\n")
        else:
            self.synthesis_results.insert(tk.END, "No synthesis results generated.")
        
        # Re-enable the button
        self.synthesize_button.configure(state="normal")
    
    def refresh_stats(self):
        """Refresh the statistics display"""
        try:
            # Clear existing items
            for item in self.stats_tree.get_children():
                self.stats_tree.delete(item)
            
            # Get current stats
            stats = self.memory_system.get_stats()
            
            # Add synthesis stats
            synthesis_node = self.stats_tree.insert("", "end", text="Synthesis Statistics", open=True)
            
            synthesis_stats = stats.get("synthesis_stats", {})
            self.stats_tree.insert(synthesis_node, "end", text="Synthesis Count", 
                                 values=(synthesis_stats.get("synthesis_count", 0),))
            
            topics = synthesis_stats.get("topics_synthesized", [])
            topics_str = ", ".join(topics[:5]) + (", ..." if len(topics) > 5 else "")
            self.stats_tree.insert(synthesis_node, "end", text="Topics Synthesized", 
                                 values=(f"{len(topics)} topics" if topics else "None",))
            
            timestamp = synthesis_stats.get("last_synthesis_timestamp", "Never")
            self.stats_tree.insert(synthesis_node, "end", text="Last Synthesis", values=(timestamp,))
            
            # Add language memory stats
            language_node = self.stats_tree.insert("", "end", text="Language Memory Statistics", open=True)
            
            language_stats = stats.get("language_memory_stats", {})
            self.stats_tree.insert(language_node, "end", text="Memory Count", 
                                 values=(language_stats.get("memory_count", 0),))
            self.stats_tree.insert(language_node, "end", text="Sentence Count", 
                                 values=(language_stats.get("sentence_count", 0),))
            
            topics = language_stats.get("topics", [])
            topics_str = ", ".join([t[0] for t in topics[:5]]) + (", ..." if len(topics) > 5 else "")
            self.stats_tree.insert(language_node, "end", text="Top Topics", 
                                 values=(topics_str if topics else "None",))
            
            # Add component stats
            components_node = self.stats_tree.insert("", "end", text="Components", open=True)
            
            component_stats = stats.get("component_stats", {})
            for component, component_data in component_stats.items():
                status = "Active" if component_data.get("active", False) else "Inactive"
                self.stats_tree.insert(components_node, "end", text=component, values=(status,))
            
            logger.info("Statistics refreshed")
        except Exception as e:
            error_msg = f"Error refreshing statistics: {str(e)}"
            logger.error(error_msg)
            messagebox.showerror("Statistics Error", error_msg)


def main():
    """Main function for running the Language Memory GUI"""
    # Create the root window
    root = tk.Tk()
    
    # Create the GUI
    app = LanguageMemoryGUI(root)
    
    # Start the main loop
    root.mainloop()


if __name__ == "__main__":
    main() 