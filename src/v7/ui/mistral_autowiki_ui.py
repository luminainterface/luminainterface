#!/usr/bin/env python3
"""
Mistral Autowiki UI

This module provides a UI for interacting with the MistralIntegration class,
allowing users to send queries, view and edit the autowiki dictionary.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import json
import threading
import logging
from typing import Dict, Any, Optional, List

# Add parent directory to path to import MistralIntegration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mistral_integration import MistralIntegration

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MistralAutowikiUI")

class MistralAutowikiUI:
    """UI for interacting with the Mistral Autowiki system."""
    
    def __init__(self, root: tk.Tk) -> None:
        """
        Initialize the UI.
        
        Args:
            root: The tkinter root window
        """
        self.root = root
        self.root.title("Mistral Autowiki Interface")
        self.root.geometry("1000x800")
        self.root.minsize(800, 600)
        
        # Initialize Mistral integration in mock mode initially
        self.mistral = MistralIntegration(
            mock_mode=True,
            learning_enabled=True,
            learning_dict_path="data/mistral_autowiki.json"
        )
        
        self.setup_ui()
    
    def setup_ui(self) -> None:
        """Set up the UI components."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook with tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Chat tab
        chat_tab = ttk.Frame(notebook)
        notebook.add(chat_tab, text="Chat")
        
        # Autowiki tab
        autowiki_tab = ttk.Frame(notebook)
        notebook.add(autowiki_tab, text="Autowiki Dictionary")
        
        # Settings tab
        settings_tab = ttk.Frame(notebook)
        notebook.add(settings_tab, text="Settings")
        
        # Set up each tab
        self._setup_chat_tab(chat_tab)
        self._setup_autowiki_tab(autowiki_tab)
        self._setup_settings_tab(settings_tab)
        
        # Status bar
        self.status_bar = ttk.Label(
            self.root, 
            text="Ready | Mock Mode: ON | Model: None", 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Update status bar
        self._update_status_bar()
    
    def _setup_chat_tab(self, parent: ttk.Frame) -> None:
        """
        Set up the chat interface tab.
        
        Args:
            parent: The parent frame
        """
        # Split chat tab into chat history and input area
        chat_frame = ttk.Frame(parent)
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Options frame at the top
        options_frame = ttk.LabelFrame(chat_frame, text="Chat Options")
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Autowiki checkbox
        self.use_autowiki_var = tk.BooleanVar(value=True)
        autowiki_check = ttk.Checkbutton(
            options_frame, 
            text="Use Autowiki Knowledge", 
            variable=self.use_autowiki_var
        )
        autowiki_check.pack(side=tk.LEFT, padx=5)
        
        # System prompt
        ttk.Label(options_frame, text="System Prompt:").pack(side=tk.LEFT, padx=(10, 0))
        self.system_prompt = tk.StringVar(value="You are an AI assistant specializing in neural networks and consciousness.")
        system_entry = ttk.Entry(options_frame, textvariable=self.system_prompt, width=50)
        system_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Parameters frame
        params_frame = ttk.LabelFrame(chat_frame, text="Model Parameters")
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Temperature
        ttk.Label(params_frame, text="Temperature:").pack(side=tk.LEFT, padx=5)
        self.temperature_var = tk.DoubleVar(value=0.7)
        temp_scale = ttk.Scale(
            params_frame, 
            from_=0.0, 
            to=1.0, 
            variable=self.temperature_var, 
            orient=tk.HORIZONTAL,
            length=100
        )
        temp_scale.pack(side=tk.LEFT, padx=5)
        ttk.Label(params_frame, textvariable=self.temperature_var).pack(side=tk.LEFT, padx=5)
        
        # Max tokens
        ttk.Label(params_frame, text="Max Tokens:").pack(side=tk.LEFT, padx=(20, 5))
        self.max_tokens_var = tk.IntVar(value=500)
        max_tokens_entry = ttk.Spinbox(
            params_frame, 
            from_=50, 
            to=4000, 
            increment=50,
            textvariable=self.max_tokens_var,
            width=5
        )
        max_tokens_entry.pack(side=tk.LEFT, padx=5)
        
        # Clear button
        clear_btn = ttk.Button(params_frame, text="Clear Chat", command=self._clear_chat)
        clear_btn.pack(side=tk.RIGHT, padx=5)
        
        # Chat history
        history_frame = ttk.LabelFrame(chat_frame, text="Chat History")
        history_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.chat_history = scrolledtext.ScrolledText(
            history_frame, 
            wrap=tk.WORD, 
            width=80, 
            height=20,
            font=("TkDefaultFont", 10)
        )
        self.chat_history.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.chat_history.config(state=tk.DISABLED)
        
        # Input area
        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.message_input = scrolledtext.ScrolledText(
            input_frame, 
            wrap=tk.WORD, 
            width=80, 
            height=4,
            font=("TkDefaultFont", 10)
        )
        self.message_input.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=(0, 5))
        
        send_button = ttk.Button(
            input_frame, 
            text="Send", 
            command=self._send_message,
            width=10
        )
        send_button.pack(side=tk.RIGHT, padx=5)
        
        # Bind Enter key to send message
        self.message_input.bind("<Return>", self._send_message)
        self.message_input.bind("<Shift-Return>", lambda e: None)  # Allow Shift+Enter for new line
    
    def _setup_autowiki_tab(self, parent: ttk.Frame) -> None:
        """
        Set up the autowiki dictionary tab.
        
        Args:
            parent: The parent frame
        """
        # Split the tab into two frames: the left for the list of entries, the right for viewing/editing
        paned_window = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left frame - list of entries
        left_frame = ttk.Frame(paned_window)
        paned_window.add(left_frame, weight=1)
        
        # Search frame
        search_frame = ttk.Frame(left_frame)
        search_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT, padx=5)
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        search_button = ttk.Button(search_frame, text="Search", command=self._search_entries)
        search_button.pack(side=tk.RIGHT, padx=5)
        
        # Topic list frame
        topics_frame = ttk.LabelFrame(left_frame, text="Autowiki Topics")
        topics_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # List of topics
        self.topics_listbox = tk.Listbox(topics_frame, selectmode=tk.SINGLE)
        self.topics_listbox.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        topics_scrollbar = ttk.Scrollbar(topics_frame, orient=tk.VERTICAL, command=self.topics_listbox.yview)
        topics_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.topics_listbox.config(yscrollcommand=topics_scrollbar.set)
        
        # Bind selection event
        self.topics_listbox.bind('<<ListboxSelect>>', self._on_topic_select)
        
        # Buttons frame
        buttons_frame = ttk.Frame(left_frame)
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        refresh_button = ttk.Button(buttons_frame, text="Refresh List", command=self._refresh_topics)
        refresh_button.pack(side=tk.LEFT, padx=5)
        
        add_button = ttk.Button(buttons_frame, text="Add New Entry", command=self._add_new_entry)
        add_button.pack(side=tk.LEFT, padx=5)
        
        delete_button = ttk.Button(buttons_frame, text="Delete Entry", command=self._delete_entry)
        delete_button.pack(side=tk.LEFT, padx=5)
        
        # Right frame - view/edit entry
        right_frame = ttk.Frame(paned_window)
        paned_window.add(right_frame, weight=2)
        
        # Entry view/edit frame
        entry_frame = ttk.LabelFrame(right_frame, text="Entry Details")
        entry_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Topic entry
        topic_frame = ttk.Frame(entry_frame)
        topic_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(topic_frame, text="Topic:").pack(side=tk.LEFT, padx=5)
        self.topic_var = tk.StringVar()
        topic_entry = ttk.Entry(topic_frame, textvariable=self.topic_var, width=50)
        topic_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Source entry
        source_frame = ttk.Frame(entry_frame)
        source_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(source_frame, text="Source:").pack(side=tk.LEFT, padx=5)
        self.source_var = tk.StringVar()
        source_entry = ttk.Entry(source_frame, textvariable=self.source_var, width=50)
        source_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Content text area
        content_frame = ttk.LabelFrame(entry_frame, text="Content")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.content_text = scrolledtext.ScrolledText(
            content_frame, 
            wrap=tk.WORD, 
            width=60, 
            height=15,
            font=("TkDefaultFont", 10)
        )
        self.content_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Save button
        save_frame = ttk.Frame(entry_frame)
        save_frame.pack(fill=tk.X, padx=5, pady=5)
        
        save_button = ttk.Button(save_frame, text="Save Changes", command=self._save_entry)
        save_button.pack(side=tk.RIGHT, padx=5)
        
        # Initial refresh of topics
        self._refresh_topics()
    
    def _setup_settings_tab(self, parent: ttk.Frame) -> None:
        """
        Set up the settings tab.
        
        Args:
            parent: The parent frame
        """
        settings_frame = ttk.Frame(parent)
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # API Settings
        api_frame = ttk.LabelFrame(settings_frame, text="API Settings")
        api_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # API Key
        api_key_frame = ttk.Frame(api_frame)
        api_key_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(api_key_frame, text="API Key:").pack(side=tk.LEFT, padx=5)
        self.api_key_var = tk.StringVar(value=os.environ.get("MISTRAL_API_KEY", ""))
        self.api_key_entry = ttk.Entry(api_key_frame, textvariable=self.api_key_var, width=40, show="*")
        self.api_key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.show_key_var = tk.BooleanVar(value=False)
        show_key_check = ttk.Checkbutton(
            api_key_frame, 
            text="Show Key", 
            variable=self.show_key_var,
            command=self._toggle_show_key
        )
        show_key_check.pack(side=tk.LEFT, padx=5)
        
        # Mock Mode
        mock_frame = ttk.Frame(api_frame)
        mock_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.mock_mode_var = tk.BooleanVar(value=True)
        mock_check = ttk.Checkbutton(
            mock_frame, 
            text="Mock Mode (No API calls)", 
            variable=self.mock_mode_var
        )
        mock_check.pack(side=tk.LEFT, padx=5)
        
        # Model selection
        model_frame = ttk.Frame(api_frame)
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar(value="mistral-medium")
        model_combo = ttk.Combobox(
            model_frame, 
            textvariable=self.model_var,
            values=["mistral-tiny", "mistral-small", "mistral-medium", "mistral-large"],
            state="readonly",
            width=20
        )
        model_combo.pack(side=tk.LEFT, padx=5)
        
        # Learning Dictionary Settings
        learning_frame = ttk.LabelFrame(settings_frame, text="Learning Dictionary Settings")
        learning_frame.pack(fill=tk.X, padx=5, pady=(15, 5))
        
        # Learning Enabled
        enable_frame = ttk.Frame(learning_frame)
        enable_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.learning_enabled_var = tk.BooleanVar(value=True)
        learning_check = ttk.Checkbutton(
            enable_frame, 
            text="Enable Autowiki Learning", 
            variable=self.learning_enabled_var
        )
        learning_check.pack(side=tk.LEFT, padx=5)
        
        # Dictionary Path
        path_frame = ttk.Frame(learning_frame)
        path_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(path_frame, text="Dictionary Path:").pack(side=tk.LEFT, padx=5)
        self.dict_path_var = tk.StringVar(value="data/mistral_autowiki.json")
        dict_path_entry = ttk.Entry(path_frame, textvariable=self.dict_path_var, width=40)
        dict_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        browse_button = ttk.Button(path_frame, text="Browse...", command=self._browse_dict_path)
        browse_button.pack(side=tk.LEFT, padx=5)
        
        # Max memory entries
        memory_frame = ttk.Frame(learning_frame)
        memory_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(memory_frame, text="Max Memory Entries:").pack(side=tk.LEFT, padx=5)
        self.max_memory_var = tk.IntVar(value=100)
        memory_entry = ttk.Spinbox(
            memory_frame, 
            from_=10, 
            to=1000, 
            increment=10,
            textvariable=self.max_memory_var,
            width=5
        )
        memory_entry.pack(side=tk.LEFT, padx=5)
        
        # Buttons section
        button_frame = ttk.Frame(settings_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=15)
        
        # Apply button
        apply_button = ttk.Button(
            button_frame, 
            text="Apply Settings",
            command=self._apply_settings,
            width=15
        )
        apply_button.pack(side=tk.RIGHT, padx=5)
        
        # Import/Export buttons
        export_button = ttk.Button(
            button_frame, 
            text="Export Dictionary",
            command=self._export_dictionary,
            width=15
        )
        export_button.pack(side=tk.LEFT, padx=5)
        
        import_button = ttk.Button(
            button_frame, 
            text="Import Dictionary",
            command=self._import_dictionary,
            width=15
        )
        import_button.pack(side=tk.LEFT, padx=5)
        
        # Usage metrics
        metrics_frame = ttk.LabelFrame(settings_frame, text="Usage Metrics")
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.metrics_text = scrolledtext.ScrolledText(
            metrics_frame, 
            wrap=tk.WORD, 
            width=60, 
            height=8,
            font=("TkDefaultFont", 10)
        )
        self.metrics_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.metrics_text.config(state=tk.DISABLED)
        
        refresh_metrics_button = ttk.Button(
            metrics_frame, 
            text="Refresh Metrics",
            command=self._refresh_metrics,
            width=15
        )
        refresh_metrics_button.pack(side=tk.RIGHT, padx=5, pady=5)
    
    def _update_status_bar(self) -> None:
        """Update the status bar with current settings."""
        mock_status = "ON" if self.mistral.mock_mode else "OFF"
        model_status = self.mistral.model or "None"
        self.status_bar.config(text=f"Ready | Mock Mode: {mock_status} | Model: {model_status}")
    
    def _send_message(self, event=None) -> None:
        """
        Send a message to the Mistral model.
        
        Args:
            event: The event that triggered this function (optional)
        """
        message = self.message_input.get("1.0", tk.END).strip()
        if not message:
            return
        
        # Disable input while processing
        self.message_input.config(state=tk.DISABLED)
        
        # Add user message to chat history
        self._add_to_chat("You", message)
        
        # Clear input
        self.message_input.delete("1.0", tk.END)
        
        # Process in a separate thread
        threading.Thread(target=self._process_message, args=(message,), daemon=True).start()
        
        # Prevent default behavior if called from an event
        return "break"
    
    def _process_message(self, message: str) -> None:
        """
        Process a message with the Mistral integration.
        
        Args:
            message: The message to process
        """
        try:
            # Get parameters from UI
            system_prompt = self.system_prompt.get()
            temperature = self.temperature_var.get()
            max_tokens = self.max_tokens_var.get()
            include_autowiki = self.use_autowiki_var.get()
            
            # Update status
            self.root.after(0, lambda: self.status_bar.config(text="Processing message..."))
            
            # Process message
            response = self.mistral.process_message(
                message=message,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                include_autowiki=include_autowiki
            )
            
            # Add response to chat history
            model_name = response.get('model', 'Assistant')
            response_text = response.get('response', 'No response received')
            
            self.root.after(0, lambda: self._add_to_chat(model_name, response_text))
            
            # Re-enable input and update status
            self.root.after(0, lambda: self.message_input.config(state=tk.NORMAL))
            self.root.after(0, self._update_status_bar)
            
            # Refresh metrics
            self.root.after(0, self._refresh_metrics)
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to process message: {str(e)}"))
            self.root.after(0, lambda: self.message_input.config(state=tk.NORMAL))
            self.root.after(0, self._update_status_bar)
    
    def _add_to_chat(self, sender: str, message: str) -> None:
        """
        Add a message to the chat history.
        
        Args:
            sender: The sender of the message
            message: The message text
        """
        self.chat_history.config(state=tk.NORMAL)
        
        # Add a separator if there are already messages
        if self.chat_history.get("1.0", tk.END).strip():
            self.chat_history.insert(tk.END, "\n\n")
        
        # Add sender with different styling
        self.chat_history.insert(tk.END, f"{sender}:\n", "sender")
        self.chat_history.insert(tk.END, f"{message}")
        
        # Scroll to the end
        self.chat_history.see(tk.END)
        self.chat_history.config(state=tk.DISABLED)
    
    def _clear_chat(self) -> None:
        """Clear the chat history."""
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.delete("1.0", tk.END)
        self.chat_history.config(state=tk.DISABLED)
    
    def _refresh_topics(self) -> None:
        """Refresh the list of autowiki topics."""
        self.topics_listbox.delete(0, tk.END)
        
        # Get all topics
        topics = self.mistral.get_all_autowiki_topics()
        
        # Add to listbox
        for topic in sorted(topics):
            self.topics_listbox.insert(tk.END, topic)
    
    def _search_entries(self) -> None:
        """Search for entries in the autowiki."""
        search_term = self.search_var.get().lower()
        if not search_term:
            self._refresh_topics()
            return
        
        self.topics_listbox.delete(0, tk.END)
        
        # Get all topics
        topics = self.mistral.get_all_autowiki_topics()
        
        # Filter by search term
        filtered_topics = [topic for topic in topics if search_term in topic.lower()]
        
        # Add to listbox
        for topic in sorted(filtered_topics):
            self.topics_listbox.insert(tk.END, topic)
    
    def _on_topic_select(self, event) -> None:
        """
        Handle topic selection from the listbox.
        
        Args:
            event: The selection event
        """
        # Get selected topic
        selection = self.topics_listbox.curselection()
        if not selection:
            return
        
        topic = self.topics_listbox.get(selection[0])
        
        # Get entry from autowiki
        entry = self.mistral.retrieve_autowiki(topic)
        if not entry:
            messagebox.showerror("Error", f"Could not find entry for topic: {topic}")
            return
        
        # Update UI
        self.topic_var.set(topic)
        self.source_var.set(entry.get('sources', [''])[0] if entry.get('sources') else '')
        
        self.content_text.delete("1.0", tk.END)
        self.content_text.insert("1.0", entry.get('content', ''))
    
    def _save_entry(self) -> None:
        """Save the current entry to the autowiki."""
        topic = self.topic_var.get().strip()
        source = self.source_var.get().strip()
        content = self.content_text.get("1.0", tk.END).strip()
        
        if not topic:
            messagebox.showerror("Error", "Topic cannot be empty")
            return
        
        if not content:
            messagebox.showerror("Error", "Content cannot be empty")
            return
        
        # Add or update entry
        success = self.mistral.add_autowiki_entry(
            topic=topic,
            content=content,
            source=source
        )
        
        if success:
            messagebox.showinfo("Success", f"Successfully saved entry: {topic}")
            # Refresh topics
            self._refresh_topics()
        else:
            messagebox.showerror("Error", f"Failed to save entry: {topic}")
    
    def _add_new_entry(self) -> None:
        """Add a new entry to the autowiki."""
        # Clear fields
        self.topic_var.set("")
        self.source_var.set("")
        self.content_text.delete("1.0", tk.END)
    
    def _delete_entry(self) -> None:
        """Delete the selected entry from the autowiki."""
        selection = self.topics_listbox.curselection()
        if not selection:
            messagebox.showerror("Error", "No topic selected")
            return
        
        topic = self.topics_listbox.get(selection[0])
        
        # Confirm deletion
        confirm = messagebox.askyesno(
            "Confirm Deletion", 
            f"Are you sure you want to delete the entry for '{topic}'?"
        )
        
        if not confirm:
            return
        
        # Delete entry
        success = self.mistral.delete_autowiki_entry(topic)
        
        if success:
            messagebox.showinfo("Success", f"Successfully deleted entry: {topic}")
            # Refresh topics
            self._refresh_topics()
            # Clear fields
            self.topic_var.set("")
            self.source_var.set("")
            self.content_text.delete("1.0", tk.END)
        else:
            messagebox.showerror("Error", f"Failed to delete entry: {topic}")
    
    def _toggle_show_key(self) -> None:
        """Toggle showing/hiding the API key."""
        if self.show_key_var.get():
            self.api_key_entry.config(show="")
        else:
            self.api_key_entry.config(show="*")
    
    def _browse_dict_path(self) -> None:
        """Browse for a dictionary file path."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir="./data",
            title="Select Dictionary File"
        )
        
        if filepath:
            self.dict_path_var.set(filepath)
    
    def _apply_settings(self) -> None:
        """Apply the current settings to the Mistral integration."""
        # Get settings from UI
        api_key = self.api_key_var.get().strip()
        mock_mode = self.mock_mode_var.get()
        model = self.model_var.get()
        learning_enabled = self.learning_enabled_var.get()
        dict_path = self.dict_path_var.get().strip()
        max_memory = self.max_memory_var.get()
        
        # Save current dictionary
        if self.mistral.learning_enabled:
            self.mistral.save_learning_dictionary()
        
        # Reinitialize Mistral integration
        try:
            self.mistral = MistralIntegration(
                api_key=api_key,
                model=model,
                mock_mode=mock_mode,
                learning_enabled=learning_enabled,
                learning_dict_path=dict_path,
                max_memory_entries=max_memory
            )
            
            # Update status bar
            self._update_status_bar()
            
            # Refresh topics
            self._refresh_topics()
            
            # Refresh metrics
            self._refresh_metrics()
            
            messagebox.showinfo("Success", "Settings applied successfully")
            
        except Exception as e:
            logger.error(f"Error applying settings: {str(e)}")
            messagebox.showerror("Error", f"Failed to apply settings: {str(e)}")
    
    def _refresh_metrics(self) -> None:
        """Refresh and display usage metrics."""
        metrics = self.mistral.get_metrics()
        
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete("1.0", tk.END)
        
        self.metrics_text.insert(tk.END, f"API Calls: {metrics['api_calls']}\n")
        self.metrics_text.insert(tk.END, f"Total Tokens Used: {metrics['tokens_used']}\n")
        self.metrics_text.insert(tk.END, f"Prompt Tokens: {metrics['tokens_prompt']}\n")
        self.metrics_text.insert(tk.END, f"Completion Tokens: {metrics['tokens_completion']}\n")
        self.metrics_text.insert(tk.END, f"Learning Dictionary Size: {metrics['learning_dict_size']} bytes\n")
        self.metrics_text.insert(tk.END, f"Autowiki Entries: {metrics['autowiki_entries']}\n")
        
        self.metrics_text.config(state=tk.DISABLED)
    
    def _export_dictionary(self) -> None:
        """Export the autowiki dictionary to a file."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir="./data",
            title="Export Dictionary"
        )
        
        if not filepath:
            return
        
        try:
            # Save current dictionary
            self.mistral.save_learning_dictionary()
            
            # Copy to new location
            with open(self.mistral.learning_dict_path, 'r') as src_file:
                dictionary_data = json.load(src_file)
            
            with open(filepath, 'w') as dst_file:
                json.dump(dictionary_data, dst_file, indent=2)
            
            messagebox.showinfo("Success", f"Dictionary exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting dictionary: {str(e)}")
            messagebox.showerror("Error", f"Failed to export dictionary: {str(e)}")
    
    def _import_dictionary(self) -> None:
        """Import an autowiki dictionary from a file."""
        filepath = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir="./data",
            title="Import Dictionary"
        )
        
        if not filepath:
            return
        
        try:
            # Copy from source
            with open(filepath, 'r') as src_file:
                dictionary_data = json.load(src_file)
            
            # Save to current path
            with open(self.mistral.learning_dict_path, 'w') as dst_file:
                json.dump(dictionary_data, dst_file, indent=2)
            
            # Reload
            self.mistral._load_learning_dictionary()
            
            # Refresh topics
            self._refresh_topics()
            
            # Refresh metrics
            self._refresh_metrics()
            
            messagebox.showinfo("Success", f"Dictionary imported from {filepath}")
            
        except Exception as e:
            logger.error(f"Error importing dictionary: {str(e)}")
            messagebox.showerror("Error", f"Failed to import dictionary: {str(e)}")


def main() -> None:
    """Run the Mistral Autowiki UI application."""
    # Create the root window
    root = tk.Tk()
    
    # Create the UI
    app = MistralAutowikiUI(root)
    
    # Start the main loop
    root.mainloop()


if __name__ == "__main__":
    main() 