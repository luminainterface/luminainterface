from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QPushButton, QLineEdit, QTextEdit, QLabel,
    QScrollArea, QFrame, QSplitter, QTreeWidget,
    QTreeWidgetItem, QComboBox, QSpinBox
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QIcon, QPalette, QColor
import logging
from typing import Dict, List, Optional
from .article_manager import ArticleManager
from .content_generator import ContentGenerator
from .auto_learning import AutoLearningEngine

class ModernFrame(QFrame):
    """Modern styled frame"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setStyleSheet("""
            ModernFrame {
                background-color: #ffffff;
                border-radius: 10px;
                border: 1px solid #e0e0e0;
            }
        """)

class ArticleEditor(ModernFrame):
    """Article editing widget"""
    article_saved = Signal(dict)  # Emitted when article is saved
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title_layout = QHBoxLayout()
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Article Title")
        self.title_edit.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                font-size: 16px;
            }
        """)
        title_layout.addWidget(self.title_edit)
        
        # Category
        self.category_combo = QComboBox()
        self.category_combo.addItems(["Technical", "Process", "Documentation"])
        self.category_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
            }
        """)
        title_layout.addWidget(self.category_combo)
        
        layout.addLayout(title_layout)
        
        # Content
        self.content_edit = QTextEdit()
        self.content_edit.setStyleSheet("""
            QTextEdit {
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
            }
        """)
        layout.addWidget(self.content_edit)
        
        # Toolbar
        toolbar = QHBoxLayout()
        
        self.save_btn = QPushButton("Save")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.save_btn.clicked.connect(self.save_article)
        
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 5px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.generate_btn.clicked.connect(self.generate_content)
        
        toolbar.addWidget(self.save_btn)
        toolbar.addWidget(self.generate_btn)
        toolbar.addStretch()
        
        layout.addLayout(toolbar)
        
    def save_article(self):
        """Save the current article"""
        article = {
            'title': self.title_edit.text(),
            'category': self.category_combo.currentText(),
            'content': self.content_edit.toPlainText()
        }
        self.article_saved.emit(article)
        
    def generate_content(self):
        """Generate article content"""
        title = self.title_edit.text()
        category = self.category_combo.currentText()
        
        # Use content generator
        generator = ContentGenerator()
        article = generator.generate_article(
            title=title,
            category=category,
            keywords=[],  # TODO: Add keyword support
            template_type='technical' if category == 'Technical' else 'standard'
        )
        
        if article:
            self.content_edit.setPlainText(article['content'])

class ArticleList(ModernFrame):
    """Article list widget"""
    article_selected = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Search
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search articles...")
        self.search_edit.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
            }
        """)
        self.search_edit.textChanged.connect(self.filter_articles)
        layout.addWidget(self.search_edit)
        
        # Tree widget
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Title", "Category"])
        self.tree.setStyleSheet("""
            QTreeWidget {
                border: 1px solid #e0e0e0;
                border-radius: 5px;
            }
            QTreeWidget::item {
                padding: 4px;
            }
        """)
        self.tree.itemClicked.connect(self.on_article_selected)
        layout.addWidget(self.tree)
        
    def add_article(self, article: Dict):
        """Add article to the list"""
        item = QTreeWidgetItem([
            article['title'],
            article['category']
        ])
        item.setData(0, Qt.UserRole, article)
        self.tree.addTopLevelItem(item)
        
    def filter_articles(self, text: str):
        """Filter articles by search text"""
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            item.setHidden(
                text.lower() not in item.text(0).lower() and
                text.lower() not in item.text(1).lower()
            )
            
    def on_article_selected(self, item: QTreeWidgetItem, column: int):
        """Handle article selection"""
        article = item.data(0, Qt.UserRole)
        self.article_selected.emit(article)

class SuggestionPanel(ModernFrame):
    """Article improvement suggestions panel"""
    suggestion_applied = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Suggestions")
        header.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #333;
            }
        """)
        layout.addWidget(header)
        
        # Suggestions list
        self.suggestions_layout = QVBoxLayout()
        layout.addLayout(self.suggestions_layout)
        
        layout.addStretch()
        
    def set_suggestions(self, suggestions: List[Dict]):
        """Update suggestions list"""
        # Clear existing suggestions
        for i in reversed(range(self.suggestions_layout.count())):
            self.suggestions_layout.itemAt(i).widget().deleteLater()
            
        # Add new suggestions
        for suggestion in suggestions:
            suggestion_widget = self.create_suggestion_widget(suggestion)
            self.suggestions_layout.addWidget(suggestion_widget)
            
    def create_suggestion_widget(self, suggestion: Dict) -> QWidget:
        """Create widget for a single suggestion"""
        widget = QFrame()
        widget.setStyleSheet("""
            QFrame {
                background-color: #f5f5f5;
                border-radius: 5px;
                padding: 8px;
                margin: 4px;
            }
        """)
        
        layout = QVBoxLayout(widget)
        
        # Type and importance
        header = QHBoxLayout()
        type_label = QLabel(suggestion['type'].replace('_', ' ').title())
        type_label.setStyleSheet("font-weight: bold;")
        header.addWidget(type_label)
        
        importance = QLabel(suggestion['importance'])
        importance.setStyleSheet(f"""
            color: {'#f44336' if suggestion['importance'] == 'high'
                   else '#ff9800' if suggestion['importance'] == 'medium'
                   else '#4caf50'};
            font-size: 12px;
        """)
        header.addWidget(importance)
        header.addStretch()
        
        layout.addLayout(header)
        
        # Reason
        reason = QLabel(suggestion['reason'])
        reason.setWordWrap(True)
        layout.addWidget(reason)
        
        # Apply button
        if 'section' in suggestion:
            apply_btn = QPushButton("Apply")
            apply_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border-radius: 3px;
                    padding: 4px 8px;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
            """)
            apply_btn.clicked.connect(
                lambda: self.suggestion_applied.emit(suggestion)
            )
            layout.addWidget(apply_btn)
            
        return widget

class AutoWikiUI(QWidget):
    """Main AutoWiki UI"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.article_manager = ArticleManager()
        self.content_generator = ContentGenerator()
        self.learning_engine = AutoLearningEngine()
        
        self.setup_ui()
        self.load_articles()
        
    def setup_ui(self):
        layout = QHBoxLayout(self)
        
        # Article list (left panel)
        self.article_list = ArticleList()
        self.article_list.article_selected.connect(self.load_article)
        
        # Editor and suggestions (right panel)
        right_panel = QVBoxLayout()
        
        self.editor = ArticleEditor()
        self.editor.article_saved.connect(self.save_article)
        
        self.suggestions = SuggestionPanel()
        self.suggestions.suggestion_applied.connect(self.apply_suggestion)
        
        right_panel.addWidget(self.editor, stretch=2)
        right_panel.addWidget(self.suggestions, stretch=1)
        
        # Add panels to main layout
        layout.addWidget(self.article_list)
        container = QWidget()
        container.setLayout(right_panel)
        layout.addWidget(container)
        
        # Set layout proportions
        layout.setStretch(0, 1)  # Article list
        layout.setStretch(1, 2)  # Editor + Suggestions
        
    def load_articles(self):
        """Load all articles"""
        articles = self.article_manager.get_all_articles()
        for article in articles:
            self.article_list.add_article(article)
            
    def load_article(self, article: Dict):
        """Load article into editor"""
        self.editor.title_edit.setText(article['title'])
        self.editor.category_combo.setCurrentText(article['category'])
        self.editor.content_edit.setPlainText(article['content'])
        
        # Update suggestions
        suggestions = self.content_generator.suggest_improvements(
            article['content']
        )
        self.suggestions.set_suggestions(suggestions)
        
    def save_article(self, article: Dict):
        """Save article and update UI"""
        saved = self.article_manager.save_article(article)
        if saved:
            self.article_list.add_article(article)
            
    def apply_suggestion(self, suggestion: Dict):
        """Apply a suggestion to the current article"""
        content = self.editor.content_edit.toPlainText()
        
        if suggestion['type'] == 'expand_section':
            # Generate additional content for the section
            section = suggestion['section']
            new_content = self.content_generator.generate_section(
                title=self.editor.title_edit.text(),
                section=section,
                keywords=[],
                elements=['details', 'examples'],
                length='medium'
            )
            
            # TODO: Insert new content at appropriate section
            self.editor.content_edit.setPlainText(content + "\n\n" + new_content)
            
        # Update suggestions after applying
        suggestions = self.content_generator.suggest_improvements(
            self.editor.content_edit.toPlainText()
        )
        self.suggestions.set_suggestions(suggestions) 