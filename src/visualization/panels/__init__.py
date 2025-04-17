#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LUMINA V7 Dashboard Panels
===========================

This package contains various visualization panels for the LUMINA V7 Dashboard.
"""

from src.visualization.panels.base_panel import BasePanel
from src.visualization.panels.neural_activity_panel import NeuralActivityPanel
from src.visualization.panels.language_processing_panel import LanguageProcessingPanel
from src.visualization.panels.system_metrics_panel import SystemMetricsPanel

__all__ = [
    'BasePanel',
    'NeuralActivityPanel',
    'LanguageProcessingPanel',
    'SystemMetricsPanel'
] 