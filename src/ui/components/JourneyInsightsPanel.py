from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QComboBox, QScrollArea, QFrame, QSpinBox, QSplitter,
                             QGroupBox, QGridLayout, QSlider, QProgressBar, QTabWidget)
from PySide6.QtCore import Qt, Signal, QRectF, QSize
from PySide6.QtGui import QPainter, QBrush, QPen, QColor, QFont, QLinearGradient, QPainterPath

import math
import random

class InsightCardWidget(QWidget):
    """Widget for displaying a single insight card"""
    
    card_clicked = Signal(dict)  # Signal emitted when a card is clicked
    
    def __init__(self, insight_data, parent=None):
        super().__init__(parent)
        self.insight_data = insight_data
        self.setMinimumSize(250, 150)
        self.setMaximumHeight(200)
        self.is_hovering = False
        self.setMouseTracking(True)
        
        # Set up styling based on insight type
        self.colors = {
            "pattern": QColor(75, 150, 210),      # Blue
            "recommendation": QColor(100, 180, 120),  # Green
            "milestone": QColor(220, 180, 80),    # Gold
            "challenge": QColor(220, 110, 110),   # Red
            "opportunity": QColor(170, 120, 200)  # Purple
        }
    
    def enterEvent(self, event):
        """Handle mouse enter event"""
        self.is_hovering = True
        self.update()
    
    def leaveEvent(self, event):
        """Handle mouse leave event"""
        self.is_hovering = False
        self.update()
    
    def mousePressEvent(self, event):
        """Handle mouse click event"""
        self.card_clicked.emit(self.insight_data)
    
    def paintEvent(self, event):
        """Paint the insight card"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Card dimensions
        width = self.width()
        height = self.height()
        
        # Get card color based on insight type
        insight_type = self.insight_data.get("type", "pattern")
        base_color = self.colors.get(insight_type, QColor(150, 150, 150))
        
        # Lighter color for hover state
        if self.is_hovering:
            base_color = QColor(
                min(255, base_color.red() + 20),
                min(255, base_color.green() + 20),
                min(255, base_color.blue() + 20)
            )
        
        # Create gradient background
        gradient = QLinearGradient(0, 0, 0, height)
        gradient.setColorAt(0, base_color)
        gradient.setColorAt(1, base_color.darker(120))
        
        # Draw card background
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(gradient))
        painter.drawRoundedRect(QRectF(2, 2, width-4, height-4), 8, 8)
        
        # Draw border
        if self.is_hovering:
            painter.setPen(QPen(QColor(50, 50, 50), 2))
        else:
            painter.setPen(QPen(base_color.darker(150), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(QRectF(2, 2, width-4, height-4), 8, 8)
        
        # Draw card title
        title = self.insight_data.get("title", "Unknown Insight")
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 12, QFont.Bold))
        painter.drawText(
            12, 12,
            width - 24, 25,
            Qt.AlignLeft | Qt.AlignVCenter,
            title
        )
        
        # Draw divider line
        painter.setPen(QPen(QColor(255, 255, 255, 80), 1))
        painter.drawLine(12, 45, width - 12, 45)
        
        # Draw insight content
        content = self.insight_data.get("content", "")
        painter.setPen(QColor(240, 240, 240))
        painter.setFont(QFont("Arial", 9))
        
        # Truncate content if too long
        content_rect = QRectF(12, 55, width - 24, height - 90)
        
        # Ellide text if necessary
        painter.drawText(
            content_rect,
            Qt.AlignLeft | Qt.TextWordWrap,
            content
        )
        
        # Draw meta information at bottom
        meta = self.insight_data.get("meta", "")
        painter.setPen(QColor(255, 255, 255, 180))
        painter.setFont(QFont("Arial", 8))
        painter.drawText(
            12, height - 25,
            width - 24, 20,
            Qt.AlignLeft | Qt.AlignVCenter,
            meta
        )
        
        # Draw type badge in top-right corner
        type_text = insight_type.capitalize()
        font = QFont("Arial", 8)
        font.setBold(True)
        painter.setFont(font)
        
        # Get text width
        text_width = painter.fontMetrics().horizontalAdvance(type_text)
        badge_width = text_width + 14
        
        # Draw badge background
        badge_color = QColor(255, 255, 255, 40)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(badge_color))
        painter.drawRoundedRect(
            width - badge_width - 12, 12,
            badge_width, 20,
            10, 10
        )
        
        # Draw badge text
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(
            width - badge_width - 12, 12,
            badge_width, 20,
            Qt.AlignCenter,
            type_text
        )


class TrendVisualizationWidget(QWidget):
    """Widget for visualizing learning trends"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.trend_data = self.generate_trend_data()
    
    def generate_trend_data(self):
        """Generate sample trend data"""
        # Generate data points for multiple trends
        data = {
            "Neural Activity": {
                "color": QColor(75, 150, 210),
                "values": []
            },
            "Glyph Integration": {
                "color": QColor(100, 180, 120),
                "values": []
            },
            "Learning Rate": {
                "color": QColor(220, 180, 80),
                "values": []
            }
        }
        
        # Generate 30 days of data with some variation
        base_values = {
            "Neural Activity": 40,
            "Glyph Integration": 30,
            "Learning Rate": 50
        }
        
        # Create trend patterns
        for key in data:
            current = base_values[key]
            for i in range(30):
                # Add small random variation
                change = random.uniform(-5, 5)
                
                # Add trend patterns
                if i > 15:
                    change += 1.5  # Upward trend in latter half
                
                # Add some dips and peaks
                if i % 7 == 0:
                    change = random.uniform(-10, 10)
                
                current += change
                # Keep within bounds
                current = max(10, min(90, current))
                data[key]["values"].append(current)
        
        return data
    
    def paintEvent(self, event):
        """Paint the trend visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Draw background
        painter.fillRect(0, 0, width, height, QColor(240, 240, 240))
        
        # Draw chart area
        chart_margin = 20
        chart_height = height - chart_margin * 2
        chart_width = width - chart_margin * 2
        
        # Draw title
        painter.setPen(QColor(80, 80, 80))
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        painter.drawText(
            chart_margin, 5,
            chart_width, 20,
            Qt.AlignLeft | Qt.AlignVCenter,
            "Learning Activity Trends"
        )
        
        # Draw frame
        frame_rect = QRectF(chart_margin, chart_margin, chart_width, chart_height)
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        painter.drawRect(frame_rect)
        
        # Draw grid lines
        painter.setPen(QPen(QColor(220, 220, 220), 1, Qt.DashLine))
        
        # Horizontal grid lines
        for i in range(1, 4):
            y = chart_margin + chart_height * i / 4
            painter.drawLine(chart_margin, y, chart_margin + chart_width, y)
        
        # Vertical grid lines
        for i in range(1, 6):
            x = chart_margin + chart_width * i / 6
            painter.drawLine(x, chart_margin, x, chart_margin + chart_height)
        
        # Draw trend lines
        for trend_name, trend_data in self.trend_data.items():
            values = trend_data["values"]
            color = trend_data["color"]
            
            if not values:
                continue
                
            # Create path for trend line
            path = QPainterPath()
            
            # Starting point
            x = chart_margin
            y = chart_margin + chart_height - (values[0] / 100) * chart_height
            path.moveTo(x, y)
            
            # Add points
            points_count = len(values)
            x_step = chart_width / (points_count - 1) if points_count > 1 else 0
            
            for i in range(1, points_count):
                x = chart_margin + i * x_step
                y = chart_margin + chart_height - (values[i] / 100) * chart_height
                path.lineTo(x, y)
            
            # Draw trend line
            painter.setPen(QPen(color, 2))
            painter.drawPath(path)
            
            # Draw end point label
            last_value = values[-1]
            x = chart_margin + chart_width
            y = chart_margin + chart_height - (last_value / 100) * chart_height
            
            # Draw marker
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawEllipse(x - 4, y - 4, 8, 8)
            
            # Draw label text
            painter.setPen(color.darker(120))
            painter.setFont(QFont("Arial", 8, QFont.Bold))
            painter.drawText(
                x - 30, y - 20,
                60, 16,
                Qt.AlignRight | Qt.AlignVCenter,
                f"{trend_name}"
            )
        
        # Draw date range labels
        painter.setPen(QColor(100, 100, 100))
        painter.setFont(QFont("Arial", 8))
        
        # Start date
        painter.drawText(
            chart_margin, chart_margin + chart_height + 5,
            80, 15,
            Qt.AlignLeft | Qt.AlignTop,
            "30 days ago"
        )
        
        # End date
        painter.drawText(
            chart_margin + chart_width - 80, chart_margin + chart_height + 5,
            80, 15,
            Qt.AlignRight | Qt.AlignTop,
            "Today"
        )


class RecommendationsWidget(QWidget):
    """Widget for displaying learning recommendations"""
    
    recommendation_clicked = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
    
    def initUI(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title = QLabel("Recommended Next Steps")
        title.setStyleSheet("font-size: 12px; font-weight: bold; color: #34495E; padding: 5px;")
        layout.addWidget(title)
        
        # Recommendations container
        recommendations_container = QWidget()
        self.recommendations_layout = QVBoxLayout(recommendations_container)
        self.recommendations_layout.setContentsMargins(0, 0, 0, 0)
        self.recommendations_layout.setSpacing(5)
        
        # Add sample recommendations
        self.add_sample_recommendations()
        
        layout.addWidget(recommendations_container)
    
    def add_sample_recommendations(self):
        """Add sample recommendations"""
        recommendations = [
            {
                "title": "Explore Advanced Pattern Recognition",
                "description": "Your neural network shows potential for more complex pattern recognition. Try the advanced training module.",
                "priority": "high"
            },
            {
                "title": "Integrate Glyph Set #7",
                "description": "Your mastery of basic glyphs suggests you're ready for the advanced communication glyph set.",
                "priority": "medium"
            },
            {
                "title": "Review Recent Learning Sessions",
                "description": "Analyzing your last 5 learning sessions could provide insights for optimization.",
                "priority": "medium" 
            },
            {
                "title": "Try Multi-modal Integration Exercise",
                "description": "Combining visual and semantic processing could enhance your glyph interpretation capabilities.",
                "priority": "low"
            }
        ]
        
        for rec in recommendations:
            self.add_recommendation(rec)
    
    def add_recommendation(self, rec_data):
        """Add a single recommendation item"""
        # Create recommendation item
        item = QFrame()
        item.setFrameShape(QFrame.StyledPanel)
        item.setCursor(Qt.PointingHandCursor)
        
        # Style based on priority
        priority_colors = {
            "high": "#E74C3C",
            "medium": "#F39C12",
            "low": "#3498DB"
        }
        color = priority_colors.get(rec_data.get("priority", "medium"), "#3498DB")
        
        item.setStyleSheet(f"""
            QFrame {{
                border: 1px solid #CCCCCC;
                border-left: 5px solid {color};
                background-color: #F8F8F8;
                border-radius: 4px;
                padding: 8px;
            }}
            QFrame:hover {{
                background-color: #EFEFEF;
                border: 1px solid #AAAAAA;
                border-left: 5px solid {color};
            }}
        """)
        
        # Layout
        item_layout = QVBoxLayout(item)
        item_layout.setContentsMargins(8, 5, 8, 5)
        item_layout.setSpacing(3)
        
        # Title
        title = QLabel(rec_data.get("title", "Unknown Recommendation"))
        title.setStyleSheet("font-weight: bold; color: #2C3E50;")
        item_layout.addWidget(title)
        
        # Description
        desc = QLabel(rec_data.get("description", ""))
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #7F8C8D; font-size: 9px;")
        item_layout.addWidget(desc)
        
        # Add to layout
        self.recommendations_layout.addWidget(item)
        
        # Connect click event
        item.mousePressEvent = lambda e, data=rec_data: self.recommendation_clicked.emit(data)


class JourneyInsightsPanel(QWidget):
    """Panel for displaying insights and analysis of user's learning journey"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        self.sample_insights = self.generate_sample_insights()
    
    def initUI(self):
        """Initialize the user interface"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Header
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 5)
        
        title = QLabel("Learning Insights & Analysis")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2C3E50;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Insight type filter
        self.insight_filter = QComboBox()
        self.insight_filter.addItems([
            "All Insights",
            "Patterns",
            "Recommendations",
            "Milestones",
            "Challenges",
            "Opportunities"
        ])
        header_layout.addWidget(QLabel("Show:"))
        header_layout.addWidget(self.insight_filter)
        
        # Refresh button
        self.refresh_btn = QPushButton("Refresh")
        header_layout.addWidget(self.refresh_btn)
        
        main_layout.addWidget(header)
        
        # Main content area - split into insights and sidebar
        content_splitter = QSplitter(Qt.Horizontal)
        
        # Insights area
        insights_widget = QWidget()
        insights_layout = QVBoxLayout(insights_widget)
        insights_layout.setContentsMargins(0, 0, 0, 0)
        
        # Trends chart
        self.trends_chart = TrendVisualizationWidget()
        insights_layout.addWidget(self.trends_chart)
        
        # Insights grid
        insights_label = QLabel("Key Insights")
        insights_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #34495E; margin-top: 10px;")
        insights_layout.addWidget(insights_label)
        
        # Scrollable area for insights
        insights_scroll = QScrollArea()
        insights_scroll.setWidgetResizable(True)
        insights_scroll.setFrameShape(QFrame.NoFrame)
        
        insights_container = QWidget()
        self.insights_grid = QGridLayout(insights_container)
        self.insights_grid.setContentsMargins(0, 0, 0, 0)
        self.insights_grid.setSpacing(10)
        
        insights_scroll.setWidget(insights_container)
        insights_layout.addWidget(insights_scroll)
        
        # Add insights to grid
        self.populate_insights_grid()
        
        # Add to splitter
        content_splitter.addWidget(insights_widget)
        
        # Sidebar
        sidebar_widget = QWidget()
        sidebar_widget.setMaximumWidth(250)
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(5, 0, 0, 0)
        
        # Recommendations widget
        self.recommendations = RecommendationsWidget()
        sidebar_layout.addWidget(self.recommendations)
        
        # Stats box
        stats_group = QGroupBox("Learning Stats")
        stats_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #CCCCCC;
                border-radius: 5px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
        """)
        
        stats_layout = QGridLayout(stats_group)
        
        # Add stats
        stats = [
            {"label": "Sessions Completed:", "value": "57"},
            {"label": "Total Learning Hours:", "value": "126.5"},
            {"label": "Glyphs Mastered:", "value": "78/100"},
            {"label": "Integration Score:", "value": "76%"},
            {"label": "Breakthrough Points:", "value": "3"},
            {"label": "Learning Efficiency:", "value": "83%"}
        ]
        
        for i, stat in enumerate(stats):
            label = QLabel(stat["label"])
            label.setStyleSheet("font-size: 10px; color: #7F8C8D;")
            
            value = QLabel(stat["value"])
            value.setStyleSheet("font-size: 10px; font-weight: bold; color: #2C3E50;")
            
            stats_layout.addWidget(label, i, 0)
            stats_layout.addWidget(value, i, 1)
        
        sidebar_layout.addWidget(stats_group)
        
        # Learning path progress
        path_group = QGroupBox("Learning Path Progress")
        path_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #CCCCCC;
                border-radius: 5px;
                margin-top: 5px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
        """)
        
        path_layout = QVBoxLayout(path_group)
        
        # Add path progress bars
        paths = [
            {"name": "Neural Foundations", "progress": 85},
            {"name": "Glyph Language System", "progress": 72},
            {"name": "Integration Techniques", "progress": 58},
            {"name": "Advanced Recognition", "progress": 35}
        ]
        
        for path in paths:
            path_item = QWidget()
            path_item_layout = QVBoxLayout(path_item)
            path_item_layout.setContentsMargins(0, 0, 0, 10)
            path_item_layout.setSpacing(2)
            
            name_label = QLabel(path["name"])
            name_label.setStyleSheet("font-size: 10px; color: #34495E;")
            path_item_layout.addWidget(name_label)
            
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            progress_bar.setValue(path["progress"])
            progress_bar.setTextVisible(True)
            progress_bar.setFixedHeight(15)
            progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #CCCCCC;
                    border-radius: 3px;
                    text-align: center;
                    background-color: #F8F8F8;
                }
                QProgressBar::chunk {
                    background-color: #3498DB;
                    border-radius: 2px;
                }
            """)
            path_item_layout.addWidget(progress_bar)
            
            path_layout.addWidget(path_item)
        
        sidebar_layout.addWidget(path_group)
        
        # Add stretch to push widgets to top
        sidebar_layout.addStretch()
        
        # Add to splitter
        content_splitter.addWidget(sidebar_widget)
        
        # Set initial splitter sizes
        content_splitter.setSizes([700, 300])
        
        main_layout.addWidget(content_splitter)
        
        # Footer with status
        footer = QFrame()
        footer.setFrameShape(QFrame.StyledPanel)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(10, 5, 10, 5)
        
        self.status_label = QLabel("Insights based on your last 30 days of learning activity.")
        footer_layout.addWidget(self.status_label)
        
        main_layout.addWidget(footer)
        
        # Connect signals
        self.insight_filter.currentIndexChanged.connect(self.filter_insights)
        self.refresh_btn.clicked.connect(self.refresh_insights)
        self.recommendations.recommendation_clicked.connect(self.handle_recommendation)
    
    def generate_sample_insights(self):
        """Generate sample insights for display"""
        return [
            {
                "title": "Consistent Neural Patterns",
                "type": "pattern",
                "content": "Your neural network has formed consistent recognition patterns for symbolic glyphs in the primary category, with a 78% consistency rate across sessions.",
                "meta": "Based on 57 learning sessions"
            },
            {
                "title": "Learning Plateau Detected",
                "type": "challenge",
                "content": "Your progress in advanced glyph recognition has plateaued over the last 14 days. Consider trying the alternative training approach.",
                "meta": "Detected in recent 5 sessions"
            },
            {
                "title": "Integration Milestone Reached",
                "type": "milestone",
                "content": "You've successfully integrated neural pattern recognition with symbolic glyph interpretation at a fundamental level.",
                "meta": "Achieved on April 22, 2023"
            },
            {
                "title": "Potential for Breakthrough",
                "type": "opportunity",
                "content": "Your recent neural activity shows patterns that could lead to a breakthrough in symbolic language processing if focused training is applied.",
                "meta": "Opportunity identified yesterday"
            },
            {
                "title": "Training Frequency Benefit",
                "type": "pattern",
                "content": "Analysis shows your neural network retention improves by 35% when training sessions occur daily rather than weekly.",
                "meta": "Based on comparative analysis"
            },
            {
                "title": "Cross-Modal Integration",
                "type": "recommendation",
                "content": "Your neural network would benefit from integration exercises that combine visual and semantic processing channels.",
                "meta": "High-confidence recommendation"
            }
        ]
    
    def populate_insights_grid(self):
        """Populate the insights grid with insight cards"""
        # Clear existing insights
        while self.insights_grid.count():
            item = self.insights_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Add insights to grid (3 columns)
        for i, insight_data in enumerate(self.sample_insights):
            row = i // 3
            col = i % 3
            
            insight_card = InsightCardWidget(insight_data)
            insight_card.card_clicked.connect(self.on_insight_clicked)
            
            self.insights_grid.addWidget(insight_card, row, col)
    
    def on_insight_clicked(self, insight_data):
        """Handle insight card clicked event"""
        # In a real application, this would show detailed information about the insight
        self.status_label.setText(f"Selected insight: {insight_data['title']}")
    
    def filter_insights(self):
        """Filter insights based on selected type"""
        filter_text = self.insight_filter.currentText().lower()
        
        if filter_text == "all insights":
            # Show all insights
            for i in range(self.insights_grid.count()):
                widget = self.insights_grid.itemAt(i).widget()
                if widget:
                    widget.setVisible(True)
        else:
            # Extract type from filter text (remove trailing 's')
            filter_type = filter_text[:-1] if filter_text.endswith('s') else filter_text
            
            # Show only insights of selected type
            for i in range(self.insights_grid.count()):
                widget = self.insights_grid.itemAt(i).widget()
                if widget and isinstance(widget, InsightCardWidget):
                    insight_type = widget.insight_data.get("type", "")
                    widget.setVisible(insight_type == filter_type)
        
        self.status_label.setText(f"Showing {filter_text}.")
    
    def refresh_insights(self):
        """Refresh insights data"""
        # In a real application, this would fetch new insights data
        self.status_label.setText("Refreshing insights data...")
        
        # Simulate refresh by regenerating insights
        self.sample_insights = self.generate_sample_insights()
        self.populate_insights_grid()
        
        self.status_label.setText("Insights refreshed with latest data.")
    
    def handle_recommendation(self, rec_data):
        """Handle recommendation clicked event"""
        # In a real application, this would take action based on the recommendation
        self.status_label.setText(f"Selected recommendation: {rec_data['title']}") 