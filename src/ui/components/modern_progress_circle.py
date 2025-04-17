from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, Property, QRectF
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QConicalGradient
from ..theme import LuminaTheme

class ModernProgressCircle(QWidget):
    """Circular progress indicator with modern styling"""
    def __init__(self, parent=None, size: int = 100, stroke_width: int = None):
        super().__init__(parent)
        self._value = 0
        self.max_value = 100
        self.min_size = size
        self.stroke_width = stroke_width or int(size * 0.1)  # 10% of size by default
        self.animation = None
        self._color = QColor(LuminaTheme.COLORS['accent'])
        self._text_format = "{value}%"
        self._start_color = QColor(LuminaTheme.COLORS['accent'])
        self.setMinimumSize(self.min_size, self.min_size)
        
        # Setup animation
        self._setup_animation()
        
    def _setup_animation(self):
        """Setup value animation"""
        self.animation = QPropertyAnimation(self, b"value")
        self.animation.setDuration(300)  # 300ms animation
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        
    @Property(float)
    def value(self):
        return self._value
        
    @value.setter
    def value(self, val):
        self._value = max(0, min(val, self.max_value))
        self.update()
        
    def setValue(self, value: float, animate: bool = True):
        """Set progress value with optional animation"""
        if animate and hasattr(self, 'animation'):
            self.animation.stop()
            self.animation.setStartValue(self._value)
            self.animation.setEndValue(value)
            self.animation.start()
        else:
            self.value = value
            
    def setMaxValue(self, max_value: float):
        """Set maximum value"""
        self.max_value = max_value
        self.update()
        
    def setColor(self, color: str):
        """Set the progress color
        
        Args:
            color: Color in hex format (#RRGGBB)
        """
        self._color = QColor(color)
        self.update()
        
    @property
    def text_format(self):
        """Get the text format string"""
        return self._text_format
        
    @text_format.setter
    def text_format(self, format_str: str):
        """Set the text format string
        
        Args:
            format_str: Format string with {value} placeholder
        """
        self._text_format = format_str
        self.update()
        
    @property
    def start_color(self):
        """Get the start color"""
        return self._start_color
        
    @start_color.setter
    def start_color(self, color: str):
        """Set the start color
        
        Args:
            color: Color in hex format (#RRGGBB)
        """
        self._start_color = QColor(color)
        self.update()
        
    def paintEvent(self, event):
        """Paint the progress circle"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        try:
            # Calculate sizes
            size = min(self.width(), self.height())
            pen_width = self.stroke_width
            
            # Setup painter
            painter.setPen(QPen(QColor(LuminaTheme.COLORS['border']), pen_width))
            
            # Draw background circle
            rect = QRectF(
                pen_width/2,
                pen_width/2,
                size - pen_width,
                size - pen_width
            )
            painter.drawArc(rect, 0, 360*16)
            
            # Draw progress
            painter.setPen(QPen(self._color, pen_width))
            span = -360 * 16 * self.value / self.max_value
            painter.drawArc(rect, 90*16, span)
            
            # Draw text
            painter.setPen(QColor(LuminaTheme.COLORS['text']))
            font = LuminaTheme.FONTS.get('heading', painter.font())
            # Scale font size based on widget size, with minimum and maximum limits
            font_size = max(8, min(size // 4, 24))  # Min 8pt, max 24pt
            font.setPointSize(font_size)
            painter.setFont(font)
            
            # Format and draw text
            text = self._text_format.format(value=int(self.value))
            painter.drawText(rect, Qt.AlignCenter, text)
            
        finally:
            painter.end() 