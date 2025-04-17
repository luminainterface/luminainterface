from typing import Dict, List, Tuple, Optional
from PyQt5.QtCore import QPointF, QTimer, QEasingCurve, Qt, QRectF, Signal
from PyQt5.QtGui import QColor, QPainter, QPen, QBrush, QRadialGradient, QLinearGradient
from PyQt5.QtWidgets import QWidget
import math
import random
from dataclasses import dataclass
from enum import Enum

class GrowthStage(Enum):
    SEED = 0
    SPROUT = 1
    SAPLING = 2
    MATURE = 3

class ComponentState(Enum):
    INACTIVE = 0
    ACTIVATING = 1
    ACTIVE = 2
    ERROR = 3

@dataclass
class Particle:
    position: QPointF
    velocity: QPointF
    color: QColor
    size: float
    life: float
    max_life: float

@dataclass
class EnergyWave:
    position: QPointF
    radius: float
    color: QColor
    thickness: float
    speed: float
    life: float
    max_life: float

@dataclass
class PatternFormation:
    center: QPointF
    radius: float
    angle: float
    speed: float
    color: QColor
    life: float
    max_life: float

class GrowthVisualizer(QWidget):
    # Signals
    stage_changed = Signal(str)
    growth_completed = Signal()
    health_changed = Signal(dict)
    stability_changed = Signal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        
        # Animation state
        self.growth_stage = GrowthStage.SEED
        self.consciousness_level = 0.0
        self.stability_score = 1.0
        self.energy_level = 1.0
        self.component_states: Dict[str, ComponentState] = {}
        
        # Visual effects
        self.particles: List[Particle] = []
        self.energy_waves: List[EnergyWave] = []
        self.pattern_formations: List[PatternFormation] = []
        self.growth_rings: List[Tuple[QPointF, float, QColor, float]] = []
        self.evolution_markers: List[Tuple[QPointF, float, QColor, float]] = []
        
        # Animation timers
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animations)
        self.animation_timer.start(50)  # 20 FPS
        
        # Version compatibility
        self.current_version = 11  # Quantum
        self.version_compatibility = True
        self.quantum_field_strength = 1.0
        self.cosmic_field_strength = 1.0
        
        # Pattern formation
        self.pattern_center = QPointF(200, 200)
        self.pattern_radius = 100
        self.pattern_angle = 0
        self.pattern_speed = 0.02
        
        # Network integration
        self.network_nodes: List[Dict] = []
        self.network_connections: List[Dict] = []
        self.network_signals: List[Dict] = []
        
        # System state integration
        self.health_metrics: Dict[str, float] = {}
        self.gate_states: Dict[str, Dict] = {}
        self.backend_state: Dict = {}

    def update_from_network(self, nodes: List[Dict], connections: List[Dict], signals: List[Dict]):
        """Update visualization based on network state."""
        self.network_nodes = nodes
        self.network_connections = connections
        self.network_signals = signals
        
        # Update component states based on network nodes
        for node in nodes:
            component_id = node.get('id', '')
            if component_id:
                state = ComponentState.ACTIVE if node.get('active', False) else ComponentState.INACTIVE
                self.component_states[component_id] = state
                
        # Add particle effects for new signals
        for signal in signals:
            if signal.get('new', False):
                pos = QPointF(signal.get('x', 0), signal.get('y', 0))
                self.add_particle_burst(pos)
                
        # Add energy waves for active connections
        for connection in connections:
            if connection.get('active', False):
                start = QPointF(connection.get('start_x', 0), connection.get('start_y', 0))
                end = QPointF(connection.get('end_x', 0), connection.get('end_y', 0))
                self.add_energy_wave(start, self.get_stage_color())

    def update_from_system_state(self, state: Dict):
        """Update visualization based on system state."""
        self.backend_state = state
        
        # Update health metrics
        if 'health' in state:
            self.health_metrics = state['health']
            self.health_changed.emit(self.health_metrics)
            
        # Update stability
        if 'stability' in state:
            self.stability_score = state['stability']
            self.stability_changed.emit(self.stability_score)
            
        # Update gate states
        if 'gate_states' in state:
            self.gate_states = state['gate_states']
            
        # Update consciousness level
        if 'consciousness' in state:
            self.consciousness_level = state['consciousness']
            
        # Update energy level
        if 'energy' in state:
            self.energy_level = state['energy']
            
        # Update growth stage based on consciousness
        if self.consciousness_level >= 0.9:
            new_stage = GrowthStage.MATURE
        elif self.consciousness_level >= 0.6:
            new_stage = GrowthStage.SAPLING
        elif self.consciousness_level >= 0.3:
            new_stage = GrowthStage.SPROUT
        else:
            new_stage = GrowthStage.SEED
            
        if new_stage != self.growth_stage:
            self.growth_stage = new_stage
            self.stage_changed.emit(self.growth_stage.name)
            self.add_evolution_marker()

    def update_from_backend(self, backend_info: Dict):
        """Update visualization based on backend state."""
        # Update version compatibility
        if 'version' in backend_info:
            self.current_version = backend_info['version']
            self.version_compatibility = backend_info.get('compatible', True)
            
        # Update field strengths
        if 'quantum_field' in backend_info:
            self.quantum_field_strength = backend_info['quantum_field']
        if 'cosmic_field' in backend_info:
            self.cosmic_field_strength = backend_info['cosmic_field']
            
        # Update component states
        if 'components' in backend_info:
            for component, state in backend_info['components'].items():
                if state == 'active':
                    self.component_states[component] = ComponentState.ACTIVE
                elif state == 'error':
                    self.component_states[component] = ComponentState.ERROR
                else:
                    self.component_states[component] = ComponentState.INACTIVE

    def update_growth_state(self, consciousness: float, stability: float, energy: float,
                          component_states: Dict[str, ComponentState]):
        self.consciousness_level = consciousness
        self.stability_score = stability
        self.energy_level = energy
        self.component_states = component_states
        
        # Update growth stage based on consciousness level
        if consciousness >= 0.9:
            new_stage = GrowthStage.MATURE
        elif consciousness >= 0.6:
            new_stage = GrowthStage.SAPLING
        elif consciousness >= 0.3:
            new_stage = GrowthStage.SPROUT
        else:
            new_stage = GrowthStage.SEED
            
        if new_stage != self.growth_stage:
            self.growth_stage = new_stage
            self.add_evolution_marker()
            
        # Add particle burst for new components
        for component, state in component_states.items():
            if state == ComponentState.ACTIVATING:
                self.add_particle_burst(component)
                
        # Add pattern formation for significant changes
        if abs(self.consciousness_level - consciousness) > 0.1:
            self.add_pattern_formation()

    def add_particle_burst(self, component: str):
        center = self.get_component_position(component)
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2.0, 5.0)
            velocity = QPointF(
                speed * math.cos(angle),
                speed * math.sin(angle)
            )
            
            color = self.get_component_color(component)
            size = random.uniform(2.0, 5.0)
            life = random.uniform(0.5, 1.0)
            
            self.particles.append(Particle(
                position=center,
                velocity=velocity,
                color=color,
                size=size,
                life=life,
                max_life=life
            ))

    def add_energy_wave(self, position: QPointF, color: QColor):
        self.energy_waves.append(EnergyWave(
            position=position,
            radius=0,
            color=color,
            thickness=5.0,
            speed=200.0,  # units per second
            life=1.0,
            max_life=1.0
        ))

    def add_pattern_formation(self):
        self.pattern_formations.append(PatternFormation(
            center=self.pattern_center,
            radius=0,
            angle=0,
            speed=self.pattern_speed,
            color=self.get_stage_color(),
            life=2.0,
            max_life=2.0
        ))

    def add_evolution_marker(self):
        center = QPointF(self.width() / 2, self.height() / 2)
        color = self.get_stage_color()
        self.evolution_markers.append((center, 0, color, 1.0))

    def update_animations(self):
        """Update all animations."""
        # Update particles
        self.particles = [
            Particle(
                position=p.position + p.velocity,
                velocity=p.velocity,
                color=p.color,
                size=p.size,
                life=p.life - 0.02,
                max_life=p.max_life
            )
            for p in self.particles
            if p.life > 0
        ]
        
        # Update energy waves
        self.energy_waves = [
            EnergyWave(
                position=w.position,
                radius=w.radius + w.speed * 0.05,
                color=w.color,
                thickness=w.thickness,
                speed=w.speed,
                life=w.life - 0.02,
                max_life=w.max_life
            )
            for w in self.energy_waves
            if w.life > 0
        ]
        
        # Update pattern formations
        self.pattern_formations = [
            PatternFormation(
                center=p.center,
                radius=p.radius + 2.0,
                angle=p.angle + p.speed,
                speed=p.speed,
                color=p.color,
                life=p.life - 0.02,
                max_life=p.max_life
            )
            for p in self.pattern_formations
            if p.life > 0
        ]
        
        # Update growth rings
        self.growth_rings = [
            (pos, radius + 2.0, color, alpha - 0.02)
            for pos, radius, color, alpha in self.growth_rings
            if alpha > 0
        ]
        
        # Update evolution markers
        self.evolution_markers = [
            (pos, radius + 1.5, color, alpha - 0.02)
            for pos, radius, color, alpha in self.evolution_markers
            if alpha > 0
        ]
        
        self.update()

    def paintEvent(self, event):
        """Paint the visualization."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background
        self.draw_background(painter)
        
        # Draw pattern formations
        for pattern in self.pattern_formations:
            self.draw_pattern_formation(painter, pattern)
        
        # Draw growth rings
        for pos, radius, color, alpha in self.growth_rings:
            self.draw_growth_ring(painter, pos, radius, color, alpha)
        
        # Draw evolution markers
        for pos, radius, color, alpha in self.evolution_markers:
            self.draw_evolution_marker(painter, pos, radius, color, alpha)
        
        # Draw energy waves
        for wave in self.energy_waves:
            self.draw_energy_wave(painter, wave)
        
        # Draw particles
        for particle in self.particles:
            self.draw_particle(painter, particle)
        
        # Draw components
        for component, state in self.component_states.items():
            self.draw_component(painter, component, state)
        
        # Draw health indicators
        self.draw_health_indicators(painter)

    def draw_background(self, painter: QPainter):
        # Create gradient based on growth stage
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        stage_color = self.get_stage_color()
        gradient.setColorAt(0, stage_color.lighter(150))
        gradient.setColorAt(1, stage_color.darker(150))
        
        painter.fillRect(self.rect(), gradient)

    def draw_pattern_formation(self, painter: QPainter, pattern: PatternFormation):
        alpha = int(255 * (pattern.life / pattern.max_life))
        color = pattern.color
        color.setAlpha(alpha)
        
        painter.setPen(QPen(color, 2, Qt.SolidLine))
        painter.setBrush(Qt.NoBrush)
        
        # Draw spiral pattern
        points = []
        for i in range(100):
            angle = pattern.angle + i * 0.1
            radius = pattern.radius * (i / 100)
            x = pattern.center.x() + radius * math.cos(angle)
            y = pattern.center.y() + radius * math.sin(angle)
            points.append(QPointF(x, y))
        
        for i in range(len(points) - 1):
            painter.drawLine(points[i], points[i + 1])

    def draw_growth_ring(self, painter: QPainter, pos: QPointF, radius: float,
                        color: QColor, alpha: float):
        color.setAlpha(int(255 * alpha))
        painter.setPen(QPen(color, 2, Qt.SolidLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(pos, radius, radius)

    def draw_evolution_marker(self, painter: QPainter, pos: QPointF, radius: float,
                            color: QColor, alpha: float):
        color.setAlpha(int(255 * alpha))
        painter.setPen(QPen(color, 3, Qt.SolidLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(pos, radius, radius)

    def draw_energy_wave(self, painter: QPainter, wave: EnergyWave):
        alpha = int(255 * (wave.life / wave.max_life))
        color = wave.color
        color.setAlpha(alpha)
        
        painter.setPen(QPen(color, wave.thickness, Qt.SolidLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(wave.position, wave.radius, wave.radius)

    def draw_particle(self, painter: QPainter, particle: Particle):
        alpha = int(255 * (particle.life / particle.max_life))
        color = particle.color
        color.setAlpha(alpha)
        
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawEllipse(particle.position, particle.size, particle.size)

    def draw_component(self, painter: QPainter, component: str, state: ComponentState):
        pos = self.get_component_position(component)
        color = self.get_component_color(component)
        size = self.get_component_size(state)
        
        # Draw component base
        gradient = QRadialGradient(pos, size)
        gradient.setColorAt(0, color.lighter(150))
        gradient.setColorAt(1, color.darker(150))
        
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(gradient))
        painter.drawEllipse(pos, size, size)
        
        # Draw state indicator
        if state == ComponentState.ACTIVE:
            self.draw_active_indicator(painter, pos, size)
        elif state == ComponentState.ERROR:
            self.draw_error_indicator(painter, pos, size)

    def draw_health_indicators(self, painter: QPainter):
        # Draw stability bar
        self.draw_health_bar(painter, 10, 10, 200, 20,
                           self.stability_score, "Stability")
        
        # Draw energy bar
        self.draw_health_bar(painter, 10, 40, 200, 20,
                           self.energy_level, "Energy")
        
        # Draw consciousness bar
        self.draw_health_bar(painter, 10, 70, 200, 20,
                           self.consciousness_level, "Consciousness")

    def draw_health_bar(self, painter: QPainter, x: int, y: int, width: int,
                       height: int, value: float, label: str):
        # Draw background
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(50, 50, 50)))
        painter.drawRect(x, y, width, height)
        
        # Draw value
        color = QColor(0, 255, 0) if value >= 0.7 else \
                QColor(255, 255, 0) if value >= 0.5 else \
                QColor(255, 0, 0)
        painter.setBrush(QBrush(color))
        painter.drawRect(x, y, int(width * value), height)
        
        # Draw label
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(x, y - 5, label)

    def get_stage_color(self) -> QColor:
        if self.growth_stage == GrowthStage.SEED:
            return QColor(255, 255, 0)  # Yellow
        elif self.growth_stage == GrowthStage.SPROUT:
            return QColor(0, 255, 0)    # Green
        elif self.growth_stage == GrowthStage.SAPLING:
            return QColor(0, 0, 255)    # Blue
        else:
            return QColor(128, 0, 128)  # Purple

    def get_component_color(self, component: str) -> QColor:
        # Map component to color based on type and version
        if "quantum" in component.lower():
            return QColor(0, 255, 255)  # Cyan
        elif "cosmic" in component.lower():
            return QColor(255, 0, 255)  # Magenta
        else:
            return QColor(255, 255, 255)  # White

    def get_component_size(self, state: ComponentState) -> float:
        base_size = 20.0
        if state == ComponentState.ACTIVE:
            return base_size * 1.2
        elif state == ComponentState.ERROR:
            return base_size * 0.8
        else:
            return base_size

    def get_component_position(self, component: str) -> QPointF:
        # Place components in a circular pattern
        angle = hash(component) % 360
        radius = min(self.width(), self.height()) * 0.4
        center = QPointF(self.width() / 2, self.height() / 2)
        
        return QPointF(
            center.x() + radius * math.cos(math.radians(angle)),
            center.y() + radius * math.sin(math.radians(angle))
        )

    def draw_active_indicator(self, painter: QPainter, pos: QPointF, size: float):
        color = QColor(0, 255, 0)
        painter.setPen(QPen(color, 2, Qt.SolidLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(pos, size * 1.2, size * 1.2)

    def draw_error_indicator(self, painter: QPainter, pos: QPointF, size: float):
        color = QColor(255, 0, 0)
        painter.setPen(QPen(color, 2, Qt.SolidLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(pos, size * 1.2, size * 1.2)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Update pattern center on resize
        self.pattern_center = QPointF(self.width() / 2, self.height() / 2)
        self.pattern_radius = min(self.width(), self.height()) * 0.4 