from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QComboBox, QScrollArea, QFrame, QSpinBox, QSplitter,
                             QGroupBox, QGridLayout, QSlider, QProgressBar, QTabWidget)
from PySide6.QtCore import Qt, Signal, QRectF, QPoint, QPointF, QSize
from PySide6.QtGui import QPainter, QBrush, QPen, QColor, QFont, QLinearGradient, QPainterPath, QPolygonF

import math
import random
from datetime import datetime, timedelta

class JourneyNodeWidget(QWidget):
    """Widget for visualizing a single journey node"""
    
    node_clicked = Signal(dict)  # Signal emitted when a node is clicked
    
    def __init__(self, node_data, parent=None):
        super().__init__(parent)
        self.node_data = node_data
        self.setMinimumSize(180, 100)
        self.setMaximumSize(220, 120)
        self.is_hovering = False
        self.setMouseTracking(True)
        
        # Node color based on type
        self.colors = {
            "training": QColor(100, 180, 220),       # Blue
            "exploration": QColor(100, 200, 120),    # Green
            "integration": QColor(180, 120, 200),    # Purple
            "milestone": QColor(220, 180, 80),       # Gold
            "breakthrough": QColor(220, 120, 100)    # Red
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
        self.node_clicked.emit(self.node_data)
    
    def paintEvent(self, event):
        """Paint the journey node"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Node dimensions
        width = self.width()
        height = self.height()
        
        # Get node color
        node_type = self.node_data.get("type", "exploration")
        base_color = self.colors.get(node_type, QColor(150, 150, 150))
        
        # Lighter color for hover state
        if self.is_hovering:
            base_color = QColor(
                min(255, base_color.red() + 30),
                min(255, base_color.green() + 30),
                min(255, base_color.blue() + 30)
            )
        
        # Create gradient background
        gradient = QLinearGradient(0, 0, width, height)
        gradient.setColorAt(0, base_color.lighter(130))
        gradient.setColorAt(1, base_color)
        
        # Draw card background
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(gradient))
        painter.drawRoundedRect(QRectF(2, 2, width-4, height-4), 10, 10)
        
        # Draw border
        border_color = base_color.darker(130)
        if self.is_hovering:
            border_color = QColor(50, 50, 50)
        painter.setPen(QPen(border_color, 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(QRectF(2, 2, width-4, height-4), 10, 10)
        
        # Draw node icon based on type
        self.draw_node_icon(painter, width, height, node_type, base_color)
        
        # Draw node title
        title = self.node_data.get("title", "Unknown Node")
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        painter.drawText(
            10, 10,
            width - 20, 25,
            Qt.AlignLeft | Qt.AlignVCenter,
            title
        )
        
        # Draw timestamp
        timestamp = self.node_data.get("timestamp", "Unknown date")
        painter.setPen(QColor(230, 230, 230))
        painter.setFont(QFont("Arial", 8))
        painter.drawText(
            10, 35,
            width - 20, 20,
            Qt.AlignLeft | Qt.AlignVCenter,
            timestamp
        )
        
        # Draw description
        description = self.node_data.get("description", "")
        if description:
            painter.setPen(QColor(240, 240, 240))
            painter.setFont(QFont("Arial", 9))
            
            # Truncate description if too long
            if len(description) > 70:
                description = description[:67] + "..."
                
            painter.drawText(
                10, 55,
                width - 20, height - 65,
                Qt.AlignLeft | Qt.TextWordWrap,
                description
            )
    
    def draw_node_icon(self, painter, width, height, node_type, base_color):
        """Draw an icon in the top-right corner based on node type"""
        icon_size = 20
        icon_x = width - icon_size - 10
        icon_y = 10
        
        # Background circle for icon
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 180))
        painter.drawEllipse(icon_x, icon_y, icon_size, icon_size)
        
        # Draw icon based on type
        painter.setPen(QPen(base_color.darker(120), 2))
        if node_type == "training":
            # Draw a gear icon
            center_x = icon_x + icon_size/2
            center_y = icon_y + icon_size/2
            outer_radius = icon_size/2 - 2
            inner_radius = outer_radius * 0.6
            teeth = 8
            
            for i in range(teeth * 2):
                angle = 2 * math.pi * i / (teeth * 2)
                radius = outer_radius if i % 2 == 0 else inner_radius
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                
                if i == 0:
                    path = QPainterPath(QPointF(x, y))
                else:
                    path.lineTo(QPointF(x, y))
            
            path.closeSubpath()
            painter.drawPath(path)
            
            # Draw center circle
            painter.setBrush(QBrush(base_color.darker(120)))
            painter.drawEllipse(
                center_x - inner_radius/2,
                center_y - inner_radius/2,
                inner_radius,
                inner_radius
            )
            
        elif node_type == "exploration":
            # Draw a compass/explore icon
            center_x = icon_x + icon_size/2
            center_y = icon_y + icon_size/2
            radius = icon_size/2 - 2
            
            # Draw circle
            painter.drawEllipse(
                center_x - radius,
                center_y - radius,
                radius * 2,
                radius * 2
            )
            
            # Draw compass points
            painter.drawLine(
                center_x, center_y - radius + 2,
                center_x, center_y + radius - 2
            )
            painter.drawLine(
                center_x - radius + 2, center_y,
                center_x + radius - 2, center_y
            )
            
            # Draw N indicator
            painter.setBrush(QBrush(base_color.darker(120)))
            pointer = QPolygonF()
            pointer.append(QPointF(center_x, center_y - radius + 2))
            pointer.append(QPointF(center_x - 3, center_y - radius + 8))
            pointer.append(QPointF(center_x + 3, center_y - radius + 8))
            painter.drawPolygon(pointer)
            
        elif node_type == "integration":
            # Draw connection/integration icon
            center_x = icon_x + icon_size/2
            center_y = icon_y + icon_size/2
            radius = icon_size/2 - 2
            
            # Draw two overlapping circles
            painter.drawEllipse(
                center_x - radius + 3,
                center_y - radius + 3,
                radius * 1.4,
                radius * 1.4
            )
            
            painter.drawEllipse(
                center_x - radius * 0.4,
                center_y - radius * 0.4,
                radius * 1.4,
                radius * 1.4
            )
            
        elif node_type == "milestone":
            # Draw flag icon
            center_x = icon_x + icon_size/2
            center_y = icon_y + icon_size/2
            
            # Draw flag pole
            painter.drawLine(
                center_x - 5, center_y - 7,
                center_x - 5, center_y + 7
            )
            
            # Draw flag
            flag = QPolygonF()
            flag.append(QPointF(center_x - 5, center_y - 7))
            flag.append(QPointF(center_x + 5, center_y - 4))
            flag.append(QPointF(center_x - 5, center_y))
            painter.setBrush(QBrush(base_color.darker(120)))
            painter.drawPolygon(flag)
            
        elif node_type == "breakthrough":
            # Draw lightbulb icon
            center_x = icon_x + icon_size/2
            center_y = icon_y + icon_size/2
            
            # Draw bulb
            painter.drawEllipse(
                center_x - 5, center_y - 8,
                10, 10
            )
            
            # Draw base
            painter.drawLine(
                center_x - 3, center_y + 2,
                center_x + 3, center_y + 2
            )
            painter.drawLine(
                center_x - 2, center_y + 2,
                center_x - 2, center_y + 6
            )
            painter.drawLine(
                center_x + 2, center_y + 2,
                center_x + 2, center_y + 6
            )
            painter.drawLine(
                center_x - 3, center_y + 6,
                center_x + 3, center_y + 6
            )
            
            # Draw rays
            painter.drawLine(
                center_x, center_y - 10,
                center_x, center_y - 12
            )
            painter.drawLine(
                center_x - 7, center_y - 3,
                center_x - 9, center_y - 5
            )
            painter.drawLine(
                center_x + 7, center_y - 3,
                center_x + 9, center_y - 5
            )


class TimelineWidget(QWidget):
    """Widget for visualizing the journey timeline"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.events = self.generate_events()
        self.selected_event_index = None
        self.setMouseTracking(True)
    
    def generate_events(self):
        """Generate sample timeline events"""
        events = []
        
        # Start date (6 months ago)
        start_date = datetime.now() - timedelta(days=180)
        
        # Training events
        for i in range(10):
            date = start_date + timedelta(days=i*5)
            events.append({
                "date": date,
                "type": "training",
                "title": f"Training Session {i+1}",
                "importance": random.randint(1, 5)
            })
        
        # Exploration events
        for i in range(8):
            date = start_date + timedelta(days=i*15 + 20)
            events.append({
                "date": date,
                "type": "exploration",
                "title": f"Knowledge Exploration {i+1}",
                "importance": random.randint(1, 3)
            })
        
        # Integration events
        for i in range(5):
            date = start_date + timedelta(days=i*30 + 45)
            events.append({
                "date": date,
                "type": "integration",
                "title": f"Pattern Integration {i+1}",
                "importance": random.randint(2, 4)
            })
        
        # Milestone events
        milestones = [
            {"day": 40, "title": "First Patterns Recognized"},
            {"day": 90, "title": "Integration Framework Completed"},
            {"day": 160, "title": "Advanced Recognition Mastered"}
        ]
        
        for milestone in milestones:
            date = start_date + timedelta(days=milestone["day"])
            events.append({
                "date": date,
                "type": "milestone",
                "title": milestone["title"],
                "importance": 4
            })
        
        # Breakthrough events
        breakthroughs = [
            {"day": 75, "title": "Pattern Recognition Breakthrough"},
            {"day": 130, "title": "Self-Modification Capabilities"}
        ]
        
        for breakthrough in breakthroughs:
            date = start_date + timedelta(days=breakthrough["day"])
            events.append({
                "date": date,
                "type": "breakthrough",
                "title": breakthrough["title"],
                "importance": 5
            })
        
        # Sort events by date
        events.sort(key=lambda x: x["date"])
        
        return events
    
    def paintEvent(self, event):
        """Paint the timeline"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Draw background
        painter.fillRect(0, 0, width, height, QColor(245, 245, 245))
        
        # If no events, draw empty state
        if not self.events:
            painter.setPen(QColor(150, 150, 150))
            painter.setFont(QFont("Arial", 10))
            painter.drawText(
                0, 0, width, height,
                Qt.AlignCenter,
                "No journey events available"
            )
            return
        
        # Calculate timeline dimensions
        timeline_y = height // 2
        timeline_start_x = 50
        timeline_end_x = width - 50
        timeline_length = timeline_end_x - timeline_start_x
        
        # Draw timeline line
        painter.setPen(QPen(QColor(180, 180, 180), 2))
        painter.drawLine(timeline_start_x, timeline_y, timeline_end_x, timeline_y)
        
        # Get date range
        start_date = min(event["date"] for event in self.events)
        end_date = max(event["date"] for event in self.events)
        date_range = (end_date - start_date).days
        if date_range == 0:  # Avoid division by zero
            date_range = 1
        
        # Draw events
        event_colors = {
            "training": QColor(100, 180, 220),       # Blue
            "exploration": QColor(100, 200, 120),    # Green
            "integration": QColor(180, 120, 200),    # Purple
            "milestone": QColor(220, 180, 80),       # Gold
            "breakthrough": QColor(220, 120, 100)    # Red
        }
        
        for i, event in enumerate(self.events):
            # Calculate position based on date
            days_from_start = (event["date"] - start_date).days
            pos_x = timeline_start_x + (days_from_start / date_range) * timeline_length
            
            # Determine if event is above or below timeline
            # Alternate for better visibility
            pos_y = timeline_y - 15 if i % 2 == 0 else timeline_y + 15
            
            # Get event properties
            event_type = event.get("type", "exploration")
            importance = event.get("importance", 3)
            event_color = event_colors.get(event_type, QColor(150, 150, 150))
            
            # Size based on importance (1-5)
            size = 6 + importance * 2
            
            # Draw connecting line to timeline
            painter.setPen(QPen(event_color, 1))
            painter.drawLine(int(pos_x), timeline_y, int(pos_x), pos_y)
            
            # Highlight selected event
            is_selected = self.selected_event_index == i
            
            # Draw event marker
            painter.setPen(QPen(QColor(50, 50, 50) if is_selected else event_color, 2))
            painter.setBrush(QBrush(event_color if is_selected else QColor(245, 245, 245)))
            
            if event_type in ["milestone", "breakthrough"]:
                # Draw diamond for important events
                diamond = QPolygonF()
                diamond.append(QPointF(pos_x, pos_y - size/2))
                diamond.append(QPointF(pos_x + size/2, pos_y))
                diamond.append(QPointF(pos_x, pos_y + size/2))
                diamond.append(QPointF(pos_x - size/2, pos_y))
                painter.drawPolygon(diamond)
            else:
                # Draw circle for regular events
                painter.drawEllipse(int(pos_x - size/2), int(pos_y - size/2), size, size)
            
            # Draw date labels for selected events
            if i % 5 == 0 or event_type in ["milestone", "breakthrough"] or is_selected:
                date_str = event["date"].strftime("%b %d")
                painter.setPen(QColor(100, 100, 100))
                painter.setFont(QFont("Arial", 8))
                
                # Position date label
                if i % 2 == 0:
                    # Above timeline
                    painter.drawText(
                        int(pos_x - 30), int(pos_y - size/2 - 20),
                        60, 20,
                        Qt.AlignCenter, date_str
                    )
                else:
                    # Below timeline
                    painter.drawText(
                        int(pos_x - 30), int(pos_y + size/2),
                        60, 20,
                        Qt.AlignCenter, date_str
                    )
            
            # Draw title tooltip for selected event
            if is_selected:
                title = event.get("title", "Unknown")
                painter.setPen(QColor(50, 50, 50))
                painter.setFont(QFont("Arial", 9, QFont.Bold))
                
                # Create tooltip background
                title_rect = painter.boundingRect(
                    int(pos_x - 75), int(pos_y - 40) if i % 2 == 1 else int(pos_y + 20),
                    150, 25,
                    Qt.AlignCenter, title
                )
                
                # Adjust based on overflow off-screen
                if title_rect.right() > width:
                    diff = title_rect.right() - width + 5
                    title_rect.moveLeft(title_rect.left() - diff)
                elif title_rect.left() < 0:
                    diff = abs(title_rect.left()) + 5
                    title_rect.moveLeft(title_rect.left() + diff)
                
                # Draw tooltip background
                tooltip_color = event_color.lighter(150)
                tooltip_color.setAlpha(230)
                painter.fillRect(title_rect, tooltip_color)
                painter.setPen(QPen(QColor(50, 50, 50), 1))
                painter.drawRect(title_rect)
                
                # Draw title text
                painter.drawText(title_rect, Qt.AlignCenter, title)
    
    def mousePressEvent(self, event):
        """Handle mouse press to select events"""
        self.selected_event_index = self.get_event_at_position(event.position())
        self.update()
        
        # Emit signal if needed
        if self.selected_event_index is not None:
            # Here you would emit a signal with the selected event
            pass
    
    def mouseMoveEvent(self, event):
        """Handle mouse movement to highlight events"""
        hover_index = self.get_event_at_position(event.position())
        if hover_index != self.selected_event_index:
            self.selected_event_index = hover_index
            self.update()
    
    def get_event_at_position(self, position):
        """Find event at the given mouse position"""
        if not self.events:
            return None
            
        # Calculate timeline dimensions
        width = self.width()
        height = self.height()
        timeline_y = height // 2
        timeline_start_x = 50
        timeline_end_x = width - 50
        timeline_length = timeline_end_x - timeline_start_x
        
        # Get date range
        start_date = min(event["date"] for event in self.events)
        end_date = max(event["date"] for event in self.events)
        date_range = (end_date - start_date).days
        if date_range == 0:
            date_range = 1
        
        # Check each event
        for i, event in enumerate(self.events):
            # Calculate position based on date
            days_from_start = (event["date"] - start_date).days
            pos_x = timeline_start_x + (days_from_start / date_range) * timeline_length
            
            # Determine if event is above or below timeline
            pos_y = timeline_y - 15 if i % 2 == 0 else timeline_y + 15
            
            # Get event importance for size
            importance = event.get("importance", 3)
            size = 6 + importance * 2  # Same size calculation as in paintEvent
            
            # Calculate distance from mouse to event center
            mouse_x, mouse_y = position.x(), position.y()
            distance = math.sqrt((mouse_x - pos_x)**2 + (mouse_y - pos_y)**2)
            
            # Check if mouse is within event area (with some padding)
            if distance <= size/2 + 5:  # 5px extra for easier selection
                return i
        
        return None


class ProgressMetricsWidget(QWidget):
    """Widget for displaying journey progress metrics"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.metrics = self.generate_metrics()
        self.setMinimumHeight(150)
    
    def generate_metrics(self):
        """Generate sample progress metrics"""
        return [
            {
                "name": "Neural Knowledge",
                "value": 68,
                "color": QColor(100, 180, 220)  # Blue
            },
            {
                "name": "Glyph Mastery",
                "value": 75,
                "color": QColor(100, 200, 120)  # Green
            },
            {
                "name": "Integration Depth",
                "value": 53,
                "color": QColor(180, 120, 200)  # Purple
            },
            {
                "name": "Learning Rate",
                "value": 82,
                "color": QColor(220, 180, 80)  # Gold
            },
            {
                "name": "Insight Generation",
                "value": 45,
                "color": QColor(220, 120, 100)  # Red
            }
        ]
    
    def paintEvent(self, event):
        """Paint the progress metrics"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Fill background
        painter.fillRect(0, 0, width, height, QColor(240, 240, 240))
        
        # Calculate layout
        metrics_count = len(self.metrics)
        spacing = 10
        metric_width = (width - (metrics_count + 1) * spacing) / metrics_count
        
        # Draw each metric
        x = spacing
        for metric in self.metrics:
            # Calculate dimensions
            bar_height = 20
            y = height / 2 - bar_height / 2
            
            # Draw metric name
            painter.setPen(QColor(80, 80, 80))
            painter.setFont(QFont("Arial", 9))
            painter.drawText(
                x, y - 25,
                metric_width, 20,
                Qt.AlignCenter, metric["name"]
            )
            
            # Draw progress bar background
            painter.fillRect(
                x, y,
                metric_width, bar_height,
                QColor(220, 220, 220)
            )
            
            # Draw progress value
            value = metric["value"]
            color = metric["color"]
            
            # Draw progress bar
            painter.fillRect(
                x, y,
                metric_width * value / 100, bar_height,
                color
            )
            
            # Draw value text
            painter.setPen(QColor(50, 50, 50))
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            painter.drawText(
                x, y + bar_height + 5,
                metric_width, 20,
                Qt.AlignCenter, f"{value}%"
            )
            
            # Move to next metric position
            x += metric_width + spacing


class JourneyVisualizationPanel(QWidget):
    """Panel for visualizing user's journey through the system"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        self.sample_journey_nodes = self.generate_sample_nodes()
    
    def initUI(self):
        """Initialize the user interface"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Header
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 5)
        
        title = QLabel("Your Learning Journey")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2C3E50;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Date range selector
        self.date_range = QComboBox()
        self.date_range.addItems([
            "Last 30 days",
            "Last 90 days",
            "Last 6 months",
            "Last year",
            "All time"
        ])
        self.date_range.setCurrentIndex(2)  # Default to 6 months
        header_layout.addWidget(QLabel("View:"))
        header_layout.addWidget(self.date_range)
        
        # Filter buttons
        self.filter_btn = QPushButton("Filter")
        header_layout.addWidget(self.filter_btn)
        
        # Export button
        self.export_btn = QPushButton("Export Journey")
        header_layout.addWidget(self.export_btn)
        
        main_layout.addWidget(header)
        
        # Progress metrics section
        self.progress_metrics = ProgressMetricsWidget()
        main_layout.addWidget(self.progress_metrics)
        
        # Timeline section
        timeline_section = QWidget()
        timeline_layout = QVBoxLayout(timeline_section)
        timeline_layout.setContentsMargins(0, 5, 0, 5)
        
        timeline_header = QLabel("Journey Timeline")
        timeline_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #34495E;")
        timeline_layout.addWidget(timeline_header)
        
        self.timeline = TimelineWidget()
        timeline_layout.addWidget(self.timeline)
        
        main_layout.addWidget(timeline_section)
        
        # Journey nodes section
        nodes_section = QWidget()
        nodes_layout = QVBoxLayout(nodes_section)
        nodes_layout.setContentsMargins(0, 5, 0, 5)
        
        nodes_header = QLabel("Key Journey Points")
        nodes_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #34495E;")
        nodes_layout.addWidget(nodes_header)
        
        # Create scrollable area for journey nodes
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Content widget for scroll area
        nodes_container = QWidget()
        self.nodes_layout = QHBoxLayout(nodes_container)
        self.nodes_layout.setContentsMargins(0, 0, 0, 0)
        self.nodes_layout.setSpacing(15)
        
        scroll_area.setWidget(nodes_container)
        nodes_layout.addWidget(scroll_area)
        
        main_layout.addWidget(nodes_section)
        
        # Footer with stats
        footer = QFrame()
        footer.setFrameShape(QFrame.StyledPanel)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(10, 5, 10, 5)
        
        self.status_label = QLabel("Journey visualization loaded. Showing data for the last 6 months.")
        footer_layout.addWidget(self.status_label)
        
        footer_layout.addStretch()
        
        stats_label = QLabel("Total sessions: 57 | Learning milestones: 3 | Breakthroughs: 2")
        stats_label.setStyleSheet("color: #3498DB;")
        footer_layout.addWidget(stats_label)
        
        main_layout.addWidget(footer)
        
        # Connect signals
        self.date_range.currentIndexChanged.connect(self.update_date_range)
        self.filter_btn.clicked.connect(self.show_filter_dialog)
        self.export_btn.clicked.connect(self.export_journey)
        
        # Add sample journey nodes
        self.create_journey_nodes()
    
    def generate_sample_nodes(self):
        """Generate sample journey nodes for display"""
        return [
            {
                "title": "Pattern Recognition Breakthrough",
                "type": "breakthrough",
                "timestamp": "March 15, 2023",
                "description": "Achieved 87% accuracy in complex glyph recognition after intensive training."
            },
            {
                "title": "Neural-Glyph Integration",
                "type": "milestone",
                "timestamp": "April 22, 2023",
                "description": "Successfully mapped neural pathways to glyph interpretations for the first time."
            },
            {
                "title": "Knowledge Base Expansion",
                "type": "training",
                "timestamp": "May 10, 2023",
                "description": "Added 15,000 new glyph patterns to the training set, expanding recognition capability."
            },
            {
                "title": "Self-modification Algorithm",
                "type": "breakthrough",
                "timestamp": "July 8, 2023",
                "description": "Developed capacity for autonomous neural pathway formation in response to new glyphs."
            },
            {
                "title": "Advanced Transformation Engine",
                "type": "integration",
                "timestamp": "August 30, 2023",
                "description": "Completed integration layer between neural networks and glyph interpretation system."
            }
        ]
    
    def create_journey_nodes(self):
        """Create and add journey node widgets"""
        # Clear existing nodes
        while self.nodes_layout.count():
            item = self.nodes_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Add new nodes
        for node_data in self.sample_journey_nodes:
            node_widget = JourneyNodeWidget(node_data)
            node_widget.node_clicked.connect(self.on_node_clicked)
            self.nodes_layout.addWidget(node_widget)
        
        # Add stretch to align nodes to the left
        self.nodes_layout.addStretch()
    
    def on_node_clicked(self, node_data):
        """Handle node clicked event"""
        # In a real application, this would show detailed information about the journey node
        self.status_label.setText(f"Selected: {node_data['title']} - {node_data['timestamp']}")
    
    def update_date_range(self):
        """Update the timeline based on selected date range"""
        date_range = self.date_range.currentText()
        self.status_label.setText(f"Updated timeline to show {date_range}")
        
        # In a real application, this would update the timeline and journey nodes
    
    def show_filter_dialog(self):
        """Show dialog to filter journey events"""
        # In a real application, this would show a filter dialog
        self.status_label.setText("Filter dialog would appear here")
    
    def export_journey(self):
        """Export journey data"""
        # In a real application, this would export journey data
        self.status_label.setText("Exporting journey data...") 