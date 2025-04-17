#!/usr/bin/env python3
"""
GrowthVisualizer

This module implements the growth visualization widget.
"""

import logging
from typing import Dict, Any, List, Optional
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QBrush
import math

logger = logging.getLogger(__name__)

class GrowthVisualizer(QWidget):
    """Growth visualization widget."""
    
    # Signals
    stage_changed = Signal(str)
    growth_completed = Signal()
    
    def __init__(self, parent=None):
        """Initialize the widget."""
        super().__init__(parent)
        self._config = {}
        self._current_stage = "SEED"
        self._stages = {
            "SEED": {"color": "#3498db", "label": "Seed Stage"},
            "SPROUT": {"color": "#2ecc71", "label": "Sprout Stage"},
            "BRANCH": {"color": "#e74c3c", "label": "Branch Stage"},
            "LEAF": {"color": "#f1c40f", "label": "Leaf Stage"},
            "FLOWER": {"color": "#9b59b6", "label": "Flower Stage"},
            "FRUIT": {"color": "#e67e22", "label": "Fruit Stage"}
        }
        self._progress = 0.0
        self._animation_timer = QTimer()
        self._animation_timer.timeout.connect(self._update_animation)
        self._animation_timer.start(16)  # ~60 FPS
        
        # Set up widget
        self.setMinimumSize(200, 200)
        
        # Create layout
        layout = QVBoxLayout(self)
        self._stage_label = QLabel(self._stages[self._current_stage]["label"])
        self._stage_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._stage_label)
        
    def set_config(self, config: Dict[str, Any]):
        """Set widget configuration."""
        self._config = config
        
    def update_stage(self, stage: str, progress: float = 0.0):
        """Update growth stage."""
        if stage in self._stages:
            self._current_stage = stage
            self._progress = progress
            self._stage_label.setText(self._stages[stage]["label"])
            self.stage_changed.emit(stage)
            self.update()
            
            if progress >= 1.0:
                self.growth_completed.emit()
                
    def _update_animation(self):
        """Update animation state."""
        self.update()
        
    def paintEvent(self, event):
        """Paint the widget."""
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Draw background
            self._draw_background(painter)
            
            # Draw growth visualization
            self._draw_growth(painter)
            
            # Draw progress
            self._draw_progress(painter)
            
        except Exception as e:
            logger.error(f"Error painting widget: {e}")
            
    def _draw_background(self, painter: QPainter):
        """Draw widget background."""
        try:
            # Get background color from config
            bg_color = QColor(self._config.get('background_color', '#1E1E1E'))
            painter.fillRect(self.rect(), bg_color)
            
        except Exception as e:
            logger.error(f"Error drawing background: {e}")
            
    def _draw_growth(self, painter: QPainter):
        """Draw growth visualization."""
        try:
            # Get current stage color
            stage_color = QColor(self._stages[self._current_stage]["color"])
            
            # Draw growth visualization based on stage
            if self._current_stage == "SEED":
                self._draw_seed(painter, stage_color)
            elif self._current_stage == "SPROUT":
                self._draw_sprout(painter, stage_color)
            elif self._current_stage == "BRANCH":
                self._draw_branch(painter, stage_color)
            elif self._current_stage == "LEAF":
                self._draw_leaf(painter, stage_color)
            elif self._current_stage == "FLOWER":
                self._draw_flower(painter, stage_color)
            elif self._current_stage == "FRUIT":
                self._draw_fruit(painter, stage_color)
                
        except Exception as e:
            logger.error(f"Error drawing growth: {e}")
            
    def _draw_seed(self, painter: QPainter, color: QColor):
        """Draw seed stage."""
        try:
            # Draw seed
            painter.setPen(QPen(color.darker(), 2))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(
                self.width()/2 - 20,
                self.height()/2 - 20,
                40, 40
            )
            
        except Exception as e:
            logger.error(f"Error drawing seed: {e}")
            
    def _draw_sprout(self, painter: QPainter, color: QColor):
        """Draw sprout stage."""
        try:
            # Draw stem
            painter.setPen(QPen(color.darker(), 3))
            painter.drawLine(
                self.width()/2,
                self.height()/2 + 20,
                self.width()/2,
                self.height()/2 - 40
            )
            
            # Draw leaves
            painter.setPen(QPen(color.darker(), 2))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(
                self.width()/2 - 15,
                self.height()/2 - 30,
                30, 20
            )
            
        except Exception as e:
            logger.error(f"Error drawing sprout: {e}")
            
    def _draw_branch(self, painter: QPainter, color: QColor):
        """Draw branch stage."""
        try:
            # Draw main stem
            painter.setPen(QPen(color.darker(), 3))
            painter.drawLine(
                self.width()/2,
                self.height()/2 + 20,
                self.width()/2,
                self.height()/2 - 60
            )
            
            # Draw branches
            painter.setPen(QPen(color.darker(), 2))
            for angle in [-30, 30]:
                painter.drawLine(
                    self.width()/2,
                    self.height()/2 - 20,
                    self.width()/2 + 40 * self._progress * math.cos(math.radians(angle)),
                    self.height()/2 - 20 - 40 * self._progress * math.sin(math.radians(angle))
                )
                
        except Exception as e:
            logger.error(f"Error drawing branch: {e}")
            
    def _draw_leaf(self, painter: QPainter, color: QColor):
        """Draw leaf stage."""
        try:
            # Draw stem and branches
            self._draw_branch(painter, color)
            
            # Draw leaves
            painter.setPen(QPen(color.darker(), 2))
            painter.setBrush(QBrush(color))
            for angle in [-30, 30]:
                x = self.width()/2 + 40 * math.cos(math.radians(angle))
                y = self.height()/2 - 20 - 40 * math.sin(math.radians(angle))
                painter.drawEllipse(x - 15, y - 10, 30, 20)
                
        except Exception as e:
            logger.error(f"Error drawing leaf: {e}")
            
    def _draw_flower(self, painter: QPainter, color: QColor):
        """Draw flower stage."""
        try:
            # Draw stem and leaves
            self._draw_leaf(painter, color)
            
            # Draw flower
            painter.setPen(QPen(color.darker(), 2))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(
                self.width()/2 - 20,
                self.height()/2 - 80,
                40, 40
            )
            
        except Exception as e:
            logger.error(f"Error drawing flower: {e}")
            
    def _draw_fruit(self, painter: QPainter, color: QColor):
        """Draw fruit stage."""
        try:
            # Draw stem, leaves, and flower
            self._draw_flower(painter, color)
            
            # Draw fruit
            painter.setPen(QPen(color.darker(), 2))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(
                self.width()/2 - 15,
                self.height()/2 - 40,
                30, 30
            )
            
        except Exception as e:
            logger.error(f"Error drawing fruit: {e}")
            
    def _draw_progress(self, painter: QPainter):
        """Draw progress bar."""
        try:
            # Draw progress bar background
            painter.setPen(QPen(Qt.gray, 1))
            painter.setBrush(QBrush(Qt.darkGray))
            painter.drawRect(
                20,
                self.height() - 30,
                self.width() - 40,
                10
            )
            
            # Draw progress
            progress_color = QColor(self._stages[self._current_stage]["color"])
            painter.setPen(QPen(progress_color.darker(), 1))
            painter.setBrush(QBrush(progress_color))
            painter.drawRect(
                20,
                self.height() - 30,
                int((self.width() - 40) * self._progress),
                10
            )
            
        except Exception as e:
            logger.error(f"Error drawing progress: {e}")
            
    def cleanup(self):
        """Clean up resources."""
        try:
            self._animation_timer.stop()
            self._animation_timer.deleteLater()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}") 