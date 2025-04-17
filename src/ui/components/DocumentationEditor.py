from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                              QComboBox, QTextEdit, QToolBar, QFrame, QFileDialog,
                              QSplitter, QTreeWidget, QTreeWidgetItem, QLineEdit,
                              QMenu, QAction, QMessageBox, QTabWidget)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon, QTextCharFormat, QColor, QFont, QTextCursor, QTextListFormat

class DocumentationEditor(QWidget):
    """Rich text editor for documentation and notes about neural networks and glyphs"""
    
    document_saved = Signal(str, str)  # Path, content
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_file_path = None
        self.is_modified = False
        self.initUI()
        
    def initUI(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Header
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 5)
        
        title = QLabel("Documentation & Notes")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2C3E50;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # File operations
        self.new_btn = QPushButton("New")
        self.new_btn.clicked.connect(self.new_document)
        header_layout.addWidget(self.new_btn)
        
        self.open_btn = QPushButton("Open")
        self.open_btn.clicked.connect(self.open_document)
        header_layout.addWidget(self.open_btn)
        
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_document)
        header_layout.addWidget(self.save_btn)
        
        main_layout.addWidget(header)
        
        # Main splitter between document tree and editor
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side - Document tree
        doc_panel = QWidget()
        doc_layout = QVBoxLayout(doc_panel)
        doc_layout.setContentsMargins(0, 0, 0, 0)
        
        # Search bar for documents
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search documents...")
        self.search_box.textChanged.connect(self.filter_documents)
        doc_layout.addWidget(self.search_box)
        
        # Document tree
        self.doc_tree = QTreeWidget()
        self.doc_tree.setHeaderLabels(["Documents"])
        self.doc_tree.itemClicked.connect(self.document_selected)
        
        # Add sample categories and documents
        self.setup_sample_documents()
        
        doc_layout.addWidget(self.doc_tree)
        
        # Right side - Editor section
        editor_panel = QWidget()
        editor_layout = QVBoxLayout(editor_panel)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        
        # Document tabs
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        
        # Create default tab
        self.create_new_tab("Untitled")
        
        editor_layout.addWidget(self.tabs)
        
        # Add panels to splitter
        splitter.addWidget(doc_panel)
        splitter.addWidget(editor_panel)
        splitter.setSizes([200, 600])  # Initial sizes
        
        main_layout.addWidget(splitter, 1)
        
        # Status bar
        status_bar = QFrame()
        status_bar.setFrameShape(QFrame.StyledPanel)
        status_bar_layout = QHBoxLayout(status_bar)
        status_bar_layout.setContentsMargins(5, 2, 5, 2)
        
        self.status_label = QLabel("Ready")
        status_bar_layout.addWidget(self.status_label)
        
        self.word_count_label = QLabel("Words: 0  Characters: 0")
        status_bar_layout.addWidget(self.word_count_label, alignment=Qt.AlignRight)
        
        main_layout.addWidget(status_bar)
        
    def setup_sample_documents(self):
        """Set up sample document categories and files"""
        # Create categories
        neural_networks = QTreeWidgetItem(self.doc_tree, ["Neural Networks"])
        glyphs = QTreeWidgetItem(self.doc_tree, ["Glyphs & Symbols"])
        experiments = QTreeWidgetItem(self.doc_tree, ["Experiments"])
        references = QTreeWidgetItem(self.doc_tree, ["References"])
        
        # Add sample documents
        QTreeWidgetItem(neural_networks, ["Network Architecture"])
        QTreeWidgetItem(neural_networks, ["Training Process"])
        QTreeWidgetItem(neural_networks, ["Optimization Techniques"])
        
        QTreeWidgetItem(glyphs, ["Glyph Design Principles"])
        QTreeWidgetItem(glyphs, ["Symbol-Neuron Mapping"])
        QTreeWidgetItem(glyphs, ["Ritual Sequences"])
        
        QTreeWidgetItem(experiments, ["Experiment #1: Basic Recognition"])
        QTreeWidgetItem(experiments, ["Experiment #2: Pattern Completion"])
        
        QTreeWidgetItem(references, ["Research Papers"])
        QTreeWidgetItem(references, ["Online Resources"])
        
        # Expand all categories
        self.doc_tree.expandAll()
        
    def create_new_tab(self, title):
        """Create a new editor tab"""
        tab_widget = QWidget()
        tab_layout = QVBoxLayout(tab_widget)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        
        # Formatting toolbar
        toolbar = QToolBar()
        
        # Font family
        font_combo = QComboBox()
        font_combo.addItems(["Arial", "Times New Roman", "Courier New", "Verdana", "Georgia"])
        font_combo.setCurrentText("Arial")
        font_combo.currentTextChanged.connect(lambda f: self.format_text("font-family", f))
        toolbar.addWidget(font_combo)
        
        # Font size
        size_combo = QComboBox()
        size_combo.addItems(["8", "9", "10", "11", "12", "14", "16", "18", "20", "24", "28", "32"])
        size_combo.setCurrentText("12")
        size_combo.currentTextChanged.connect(lambda s: self.format_text("font-size", int(s)))
        toolbar.addWidget(size_combo)
        
        toolbar.addSeparator()
        
        # Bold, italic, underline
        bold_action = QAction("B", self)
        bold_action.setCheckable(True)
        bold_action.triggered.connect(lambda c: self.format_text("bold", c))
        bold_action.setFont(QFont("Arial", 10, QFont.Bold))
        toolbar.addAction(bold_action)
        
        italic_action = QAction("I", self)
        italic_action.setCheckable(True)
        italic_action.triggered.connect(lambda c: self.format_text("italic", c))
        italic_action.setFont(QFont("Arial", 10, QFont.StyleItalic))
        toolbar.addAction(italic_action)
        
        underline_action = QAction("U", self)
        underline_action.setCheckable(True)
        underline_action.triggered.connect(lambda c: self.format_text("underline", c))
        underline_action.setFont(QFont("Arial", 10, QFont.UnderlineResolved))
        toolbar.addAction(underline_action)
        
        toolbar.addSeparator()
        
        # Text alignment
        align_left = QAction("Left", self)
        align_left.triggered.connect(lambda: self.format_text("align", Qt.AlignLeft))
        toolbar.addAction(align_left)
        
        align_center = QAction("Center", self)
        align_center.triggered.connect(lambda: self.format_text("align", Qt.AlignCenter))
        toolbar.addAction(align_center)
        
        align_right = QAction("Right", self)
        align_right.triggered.connect(lambda: self.format_text("align", Qt.AlignRight))
        toolbar.addAction(align_right)
        
        toolbar.addSeparator()
        
        # List formatting
        bullet_list = QAction("Bullet List", self)
        bullet_list.triggered.connect(lambda: self.format_text("list", "bullet"))
        toolbar.addAction(bullet_list)
        
        number_list = QAction("Number List", self)
        number_list.triggered.connect(lambda: self.format_text("list", "number"))
        toolbar.addAction(number_list)
        
        tab_layout.addWidget(toolbar)
        
        # Text editor
        editor = QTextEdit()
        editor.textChanged.connect(self.text_changed)
        tab_layout.addWidget(editor)
        
        # Add tab to tab widget
        self.tabs.addTab(tab_widget, title)
        self.tabs.setCurrentWidget(tab_widget)
        
        # Focus the editor
        editor.setFocus()
        
        return editor
        
    def get_current_editor(self):
        """Get the current active text editor"""
        current_tab = self.tabs.currentWidget()
        if current_tab:
            # Return the QTextEdit from the current tab
            return current_tab.findChild(QTextEdit)
        return None
        
    def format_text(self, format_type, value):
        """Apply formatting to the selected text"""
        editor = self.get_current_editor()
        if not editor:
            return
            
        cursor = editor.textCursor()
        
        if format_type == "font-family":
            format = QTextCharFormat()
            format.setFontFamily(value)
            cursor.mergeCharFormat(format)
            editor.mergeCurrentCharFormat(format)
            
        elif format_type == "font-size":
            format = QTextCharFormat()
            format.setFontPointSize(value)
            cursor.mergeCharFormat(format)
            editor.mergeCurrentCharFormat(format)
            
        elif format_type == "bold":
            format = QTextCharFormat()
            format.setFontWeight(QFont.Bold if value else QFont.Normal)
            cursor.mergeCharFormat(format)
            editor.mergeCurrentCharFormat(format)
            
        elif format_type == "italic":
            format = QTextCharFormat()
            format.setFontItalic(value)
            cursor.mergeCharFormat(format)
            editor.mergeCurrentCharFormat(format)
            
        elif format_type == "underline":
            format = QTextCharFormat()
            format.setFontUnderline(value)
            cursor.mergeCharFormat(format)
            editor.mergeCurrentCharFormat(format)
            
        elif format_type == "align":
            editor.setAlignment(value)
            
        elif format_type == "list":
            if value == "bullet":
                # Create bullet list
                list_format = QTextListFormat()
                list_format.setStyle(QTextListFormat.ListDisc)
                cursor.createList(list_format)
            elif value == "number":
                # Create numbered list
                list_format = QTextListFormat()
                list_format.setStyle(QTextListFormat.ListDecimal)
                cursor.createList(list_format)
    
    def text_changed(self):
        """Handle text changes in the editor"""
        editor = self.get_current_editor()
        if not editor:
            return
            
        # Update word count
        text = editor.toPlainText()
        word_count = len(text.split()) if text.strip() else 0
        char_count = len(text)
        self.word_count_label.setText(f"Words: {word_count}  Characters: {char_count}")
        
        # Mark as modified
        if not self.is_modified:
            self.is_modified = True
            current_index = self.tabs.currentIndex()
            current_text = self.tabs.tabText(current_index)
            if not current_text.endswith('*'):
                self.tabs.setTabText(current_index, current_text + '*')
    
    def new_document(self):
        """Create a new document"""
        self.create_new_tab("Untitled")
        self.status_label.setText("New document created")
        self.current_file_path = None
        self.is_modified = False
    
    def open_document(self):
        """Open an existing document"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Document", "", "Text Files (*.txt);;HTML Files (*.html);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                # Create new tab with file name
                file_name = file_path.split('/')[-1]
                editor = self.create_new_tab(file_name)
                
                # Detect if HTML or plain text
                if file_path.lower().endswith('.html'):
                    editor.setHtml(content)
                else:
                    editor.setPlainText(content)
                    
                self.current_file_path = file_path
                self.is_modified = False
                self.status_label.setText(f"Opened: {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not open file: {str(e)}")
    
    def save_document(self):
        """Save the current document"""
        editor = self.get_current_editor()
        if not editor:
            return
            
        if not self.current_file_path:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Document", "", "Text Files (*.txt);;HTML Files (*.html);;All Files (*)"
            )
            
            if not file_path:
                return
                
            self.current_file_path = file_path
        
        try:
            with open(self.current_file_path, 'w', encoding='utf-8') as file:
                if self.current_file_path.lower().endswith('.html'):
                    content = editor.toHtml()
                else:
                    content = editor.toPlainText()
                    
                file.write(content)
                
            # Update tab title
            current_index = self.tabs.currentIndex()
            file_name = self.current_file_path.split('/')[-1]
            self.tabs.setTabText(current_index, file_name)
            
            self.is_modified = False
            self.status_label.setText(f"Saved: {self.current_file_path}")
            
            # Emit signal that document was saved
            self.document_saved.emit(self.current_file_path, content)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save file: {str(e)}")
    
    def close_tab(self, index):
        """Close the specified tab"""
        # Check if modified
        current_text = self.tabs.tabText(index)
        if current_text.endswith('*'):
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "This document has unsaved changes. Save before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Save:
                # Make this the current tab and save
                self.tabs.setCurrentIndex(index)
                self.save_document()
            elif reply == QMessageBox.Cancel:
                return
                
        # Close the tab
        self.tabs.removeTab(index)
        
        # If no tabs left, create a new one
        if self.tabs.count() == 0:
            self.new_document()
    
    def filter_documents(self):
        """Filter documents in the tree view based on search text"""
        search_text = self.search_box.text().lower()
        
        # Check all items
        def filter_items(item):
            # Check this item
            should_show = False
            if search_text == "" or search_text in item.text(0).lower():
                should_show = True
            
            # Check children
            child_count = item.childCount()
            for i in range(child_count):
                child = item.child(i)
                # If any child should be shown, this item should be shown too
                if filter_items(child):
                    should_show = True
                    
            item.setHidden(not should_show)
            return should_show
            
        # Apply filter to top-level items
        for i in range(self.doc_tree.topLevelItemCount()):
            filter_items(self.doc_tree.topLevelItem(i))
    
    def document_selected(self, item, column):
        """Handle document selection from the tree view"""
        # Check if the item is a leaf (document) or a category
        if item.childCount() == 0:
            # This is a document, load it
            category = item.parent().text(0)
            document_name = item.text(0)
            
            # Create a new tab with this document's name
            self.create_new_tab(document_name)
            
            # Load sample content for the document
            editor = self.get_current_editor()
            if editor:
                editor.setHtml(self.get_sample_content(category, document_name))
                self.status_label.setText(f"Loaded: {category}/{document_name}")
    
    def get_sample_content(self, category, document_name):
        """Get sample content for the selected document"""
        if category == "Neural Networks":
            if document_name == "Network Architecture":
                return """
                <h1>Neural Network Architecture</h1>
                <p>This document describes the architecture of our neural network implementation.</p>
                <h2>Key Components</h2>
                <ul>
                    <li>Input Layer: 784 neurons (28x28 input)</li>
                    <li>Hidden Layers: 2 layers with 256 and 128 neurons respectively</li>
                    <li>Output Layer: Variable depending on classification task</li>
                    <li>Activation Function: ReLU for hidden layers, Softmax for output</li>
                </ul>
                <h2>Implementation Notes</h2>
                <p>Our implementation uses PyTorch as the backend framework, with custom layers for glyph integration.</p>
                """
            elif document_name == "Training Process":
                return """
                <h1>Neural Network Training Process</h1>
                <p>This document outlines our training methodology and best practices.</p>
                <h2>Training Steps</h2>
                <ol>
                    <li>Data preparation and normalization</li>
                    <li>Model initialization with Xavier/Glorot initialization</li>
                    <li>Training loop with batch size of 64</li>
                    <li>Validation after each epoch</li>
                    <li>Early stopping based on validation loss plateau</li>
                </ol>
                <h2>Hyperparameters</h2>
                <p>Learning rate: 0.001 with Adam optimizer<br>
                Dropout rate: 0.2<br>
                Epochs: Maximum 100 with early stopping</p>
                """
        elif category == "Glyphs & Symbols":
            if document_name == "Glyph Design Principles":
                return """
                <h1>Glyph Design Principles</h1>
                <p>Guidelines for creating effective glyphs that map well to neural representations.</p>
                <h2>Core Principles</h2>
                <ul>
                    <li>Geometric Clarity: Each glyph should have a clear geometric structure</li>
                    <li>Distinctive Features: Ensure sufficient differentiation between glyphs</li>
                    <li>Systematic Variation: Related concepts should have related visual elements</li>
                    <li>Hierarchical Structure: Primary, secondary, and tertiary elements</li>
                </ul>
                <h2>Visual Grammar</h2>
                <p>Our glyph system uses a consistent visual grammar where:</p>
                <ul>
                    <li>Circles represent concepts</li>
                    <li>Lines represent relationships</li>
                    <li>Angles represent transformations</li>
                    <li>Thickness represents importance</li>
                </ul>
                """
        
        # Default content for other documents
        return f"""
        <h1>{document_name}</h1>
        <p>This is sample content for the {document_name} document in the {category} category.</p>
        <p>Replace this with actual documentation as your project evolves.</p>
        """ 