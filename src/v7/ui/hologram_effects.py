"""
Holographic Effects for V7 Interface

Provides advanced holographic effects and particle systems for the V7 interface
"""

import math
import random
from typing import List, Tuple, Dict, Any

from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsItemGroup, QGraphicsPixmapItem,
    QGraphicsPathItem, QGraphicsRectItem
)
from PySide6.QtCore import (
    Qt, QPointF, QRectF, QTimer, QPropertyAnimation,
    QEasingCurve, Property
)
from PySide6.QtGui import (
    QPainter, QColor, QBrush, QPen, QRadialGradient, QLinearGradient,
    QPainterPath, QPixmap, QImage, QTransform
)

class HologramParticle:
    """
    Individual particle for holographic effects
    """
    
    def __init__(self, 
                 pos: QPointF, 
                 velocity: QPointF, 
                 size: float = 2.0,
                 color: QColor = QColor(0, 200, 255),
                 lifetime: float = 1.0):
        """
        Initialize a holographic particle
        
        Args:
            pos: Initial position
            velocity: Initial velocity
            size: Particle size
            color: Particle color
            lifetime: Lifetime in seconds
        """
        self.pos = pos
        self.velocity = velocity
        self.size = size
        self.color = color
        self.max_lifetime = lifetime
        self.lifetime = lifetime
        self.alive = True
    
    def update(self, dt: float = 0.016):
        """
        Update particle state
        
        Args:
            dt: Time delta in seconds
        """
        # Update position
        self.pos += self.velocity * dt
        
        # Update lifetime
        self.lifetime -= dt
        if self.lifetime <= 0:
            self.alive = False
        
        # Update size and opacity based on lifetime
        life_ratio = self.lifetime / self.max_lifetime
        
        # Particles grow initially then shrink
        if life_ratio > 0.7:
            # Growing phase
            size_factor = 0.3 + 0.7 * (1.0 - (life_ratio - 0.7) / 0.3)
        else:
            # Shrinking phase
            size_factor = life_ratio / 0.7
        
        self.size = self.size * 0.9 + self.size * 0.1 * size_factor
        
        # Update color alpha
        alpha = int(255 * life_ratio)
        self.color.setAlpha(alpha)
    
    def render(self, painter: QPainter):
        """
        Render the particle
        
        Args:
            painter: QPainter instance
        """
        if not self.alive:
            return
        
        # Create gradient for particle
        gradient = QRadialGradient(self.pos, self.size * 2)
        
        # Center color (full opacity)
        center_color = QColor(self.color)
        
        # Outer color (transparent)
        outer_color = QColor(self.color)
        outer_color.setAlpha(0)
        
        gradient.setColorAt(0, center_color)
        gradient.setColorAt(1, outer_color)
        
        # Draw particle
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(gradient))
        painter.drawEllipse(self.pos, self.size, self.size)

class ParticleSystem(QGraphicsItem):
    """
    Holographic particle system for visual effects
    """
    
    def __init__(self, parent=None):
        """Initialize the particle system"""
        super().__init__(parent)
        
        # System properties
        self.particles: List[HologramParticle] = []
        self.emitter_pos = QPointF(0, 0)
        self.max_particles = 200
        self.emission_rate = 10  # Particles per second
        self.emission_timer = 0.0
        
        # Particle properties
        self.particle_size_range = (1.0, 3.0)
        self.particle_lifetime_range = (0.5, 2.0)
        self.particle_speed_range = (10.0, 30.0)
        self.particle_color = QColor(0, 200, 255)
        self.particle_color_variation = 20
        
        # Configure system
        self.setZValue(100)  # Draw on top of other elements
    
    def boundingRect(self) -> QRectF:
        """Define the bounding rectangle"""
        # Large enough to contain all potential particles
        return QRectF(-200, -200, 400, 400)
    
    def set_emitter_position(self, pos: QPointF):
        """Set the emitter position"""
        self.emitter_pos = pos
    
    def set_particle_color(self, color: QColor):
        """Set the base particle color"""
        self.particle_color = color
    
    def update_system(self, dt: float = 0.016):
        """
        Update the particle system
        
        Args:
            dt: Time delta in seconds
        """
        # Update existing particles
        for particle in self.particles[:]:
            particle.update(dt)
            
            # Remove dead particles
            if not particle.alive:
                self.particles.remove(particle)
        
        # Emit new particles
        self.emission_timer += dt
        particles_to_emit = int(self.emission_rate * self.emission_timer)
        if particles_to_emit > 0:
            self.emission_timer -= particles_to_emit / self.emission_rate
            
            for _ in range(particles_to_emit):
                if len(self.particles) < self.max_particles:
                    self._emit_particle()
        
        # Request repaint
        self.update()
    
    def _emit_particle(self):
        """Emit a single particle"""
        # Random direction
        angle = random.random() * math.pi * 2
        speed = random.uniform(*self.particle_speed_range)
        
        # Calculate velocity
        velocity = QPointF(
            speed * math.cos(angle),
            speed * math.sin(angle)
        )
        
        # Randomize particle properties
        size = random.uniform(*self.particle_size_range)
        lifetime = random.uniform(*self.particle_lifetime_range)
        
        # Color variation
        variation = self.particle_color_variation
        color = QColor(
            max(0, min(255, self.particle_color.red() + random.randint(-variation, variation))),
            max(0, min(255, self.particle_color.green() + random.randint(-variation, variation))),
            max(0, min(255, self.particle_color.blue() + random.randint(-variation, variation))),
            self.particle_color.alpha()
        )
        
        # Create and add particle
        particle = HologramParticle(
            QPointF(self.emitter_pos),
            velocity,
            size,
            color,
            lifetime
        )
        
        self.particles.append(particle)
    
    def paint(self, painter: QPainter, option, widget):
        """Render all particles"""
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        # Draw each particle
        for particle in self.particles:
            particle.render(painter)

class HologramEffect(QGraphicsPathItem):
    """
    Holographic effect for UI elements
    
    Provides futuristic hologram-style effects for UI components
    """
    
    def __init__(self, path: QPainterPath = None, parent=None):
        """Initialize the hologram effect"""
        super().__init__(parent)
        
        # Set path if provided
        if path:
            self.setPath(path)
        
        # Effect properties
        self.glow_intensity = 0.7
        self.scan_line_pos = 0.0
        self.scan_line_speed = 0.5
        self.flicker_intensity = 0.1
        self.flicker_phase = random.random() * math.pi * 2
        
        # Set base style
        self.setPen(QPen(QColor(0, 200, 255, 150), 1.5))
        self.setBrush(QBrush(QColor(0, 100, 200, 40)))
    
    def set_glow_intensity(self, intensity: float):
        """Set glow intensity (0.0-1.0)"""
        self.glow_intensity = max(0.0, min(1.0, intensity))
        self.update()
    
    def update_effect(self, dt: float = 0.016):
        """
        Update effect animation
        
        Args:
            dt: Time delta in seconds
        """
        # Update scan line position
        self.scan_line_pos += self.scan_line_speed * dt
        if self.scan_line_pos > 1.0:
            self.scan_line_pos -= 1.0
        
        # Update flicker
        self.flicker_phase += dt * 5
        if self.flicker_phase > math.pi * 2:
            self.flicker_phase -= math.pi * 2
        
        self.update()
    
    def paint(self, painter: QPainter, option, widget):
        """Custom painting with holographic effects"""
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        # Get path bounds
        path_rect = self.path().boundingRect()
        
        # Calculate flicker effect
        flicker = 1.0 - self.flicker_intensity * (0.5 + 0.5 * math.sin(self.flicker_phase))
        
        # Draw glow effect
        if self.glow_intensity > 0.0:
            # Create enlarged path for glow
            glow_path = QPainterPath(self.path())
            
            # Create glow pen
            glow_color = QColor(0, 200, 255, int(100 * self.glow_intensity * flicker))
            glow_pen = QPen(glow_color, 4 + 2 * self.glow_intensity)
            glow_pen.setCapStyle(Qt.RoundCap)
            glow_pen.setJoinStyle(Qt.RoundJoin)
            
            painter.setPen(glow_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(glow_path)
        
        # Draw base shape with standard style
        standard_color = QColor(0, 200, 255, int(150 * flicker))
        self.setPen(QPen(standard_color, 1.5))
        
        fill_color = QColor(0, 100, 200, int(40 * flicker))
        self.setBrush(QBrush(fill_color))
        
        super().paint(painter, option, widget)
        
        # Draw scan line effect
        if path_rect.isValid():
            scan_y = path_rect.top() + self.scan_line_pos * path_rect.height()
            
            # Create scan line gradient
            scan_gradient = QLinearGradient(
                path_rect.left(), scan_y - 5,
                path_rect.left(), scan_y + 5
            )
            
            scan_gradient.setColorAt(0.0, QColor(0, 200, 255, 0))
            scan_gradient.setColorAt(0.5, QColor(0, 200, 255, 150))
            scan_gradient.setColorAt(1.0, QColor(0, 200, 255, 0))
            
            # Create scan line path
            scan_path = QPainterPath()
            scan_path.moveTo(path_rect.left(), scan_y)
            scan_path.lineTo(path_rect.right(), scan_y)
            
            painter.setPen(QPen(QBrush(scan_gradient), 3))
            painter.drawPath(scan_path)

class HolographicText(QGraphicsItem):
    """
    Holographic text effect for UI labels and display
    """
    
    def __init__(self, text: str = "", parent=None):
        """Initialize holographic text"""
        super().__init__(parent)
        
        # Text properties
        self.text = text
        self.font = QFont("Consolas", 12)
        self.text_color = QColor(0, 200, 255)
        
        # Effect properties
        self.glow_intensity = 0.7
        self.flicker_intensity = 0.1
        self.flicker_phase = random.random() * math.pi * 2
        self.character_offset = []
        
        # Initialize character offsets (for floating effect)
        self._reset_character_offsets()
    
    def _reset_character_offsets(self):
        """Initialize character vertical offsets"""
        self.character_offset = [
            random.uniform(-2.0, 2.0) for _ in range(len(self.text))
        ]
    
    def set_text(self, text: str):
        """Set the text to display"""
        if self.text != text:
            self.text = text
            self._reset_character_offsets()
            self.update()
    
    def boundingRect(self) -> QRectF:
        """Define the bounding rectangle"""
        # Estimate text size based on character count and font
        metric = self.font.pixelSize() if self.font.pixelSize() > 0 else self.font.pointSize()
        width = len(self.text) * metric * 0.7
        height = metric * 2  # Account for glow
        
        return QRectF(0, 0, width, height)
    
    def update_effect(self, dt: float = 0.016):
        """
        Update holographic text animation
        
        Args:
            dt: Time delta in seconds
        """
        # Update flicker
        self.flicker_phase += dt * 5
        if self.flicker_phase > math.pi * 2:
            self.flicker_phase -= math.pi * 2
        
        # Update character offsets (floating effect)
        for i in range(len(self.character_offset)):
            # Sin wave movement for each character
            phase = self.flicker_phase + i * 0.5
            self.character_offset[i] = math.sin(phase) * 2.0
        
        self.update()
    
    def paint(self, painter: QPainter, option, widget):
        """Custom painting with holographic text effects"""
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        
        # Calculate flicker effect
        flicker = 1.0 - self.flicker_intensity * (0.5 + 0.5 * math.sin(self.flicker_phase))
        
        # Set font
        painter.setFont(self.font)
        
        # Draw characters with individual effects
        for i, char in enumerate(self.text):
            x = i * (self.font.pixelSize() * 0.7 if self.font.pixelSize() > 0 else self.font.pointSize() * 0.7)
            y = self.character_offset[i] + self.font.pixelSize() if self.font.pixelSize() > 0 else self.font.pointSize()
            
            # Draw glow
            if self.glow_intensity > 0.0:
                glow_color = QColor(self.text_color)
                glow_color.setAlpha(int(100 * self.glow_intensity * flicker))
                
                painter.setPen(glow_color)
                # Draw multiple times for blur effect
                for offset in range(1, 4):
                    painter.drawText(x - offset, y, char)
                    painter.drawText(x + offset, y, char)
                    painter.drawText(x, y - offset, char)
                    painter.drawText(x, y + offset, char)
            
            # Draw main text
            text_color = QColor(self.text_color)
            text_color.setAlpha(int(255 * flicker))
            
            painter.setPen(text_color)
            painter.drawText(x, y, char) 