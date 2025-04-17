from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                          QPushButton, QSlider, QComboBox, QCheckBox,
                          QFrame, QGroupBox, QTabWidget, QFileDialog, QSpinBox)
from PyQt5.QtGui import QIcon, QFont, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal

class SettingsPanel(QWidget):
    """Settings panel for application configuration"""
    
    settings_saved = pyqtSignal(dict)  # Signal emitted when settings are saved
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        """Initialize the settings panel UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(20)
        
        # Header
        header = QLabel("Settings")
        header.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #3498DB;
            margin-bottom: 10px;
        """)
        main_layout.addWidget(header)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #2C3E50;
                border-radius: 5px;
                padding: 5px;
            }
            QTabBar::tab {
                background-color: #1E2C3A;
                color: #95A5A6;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                padding: 8px 12px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #2C3E50;
                color: #3498DB;
            }
            QTabBar::tab:hover:!selected {
                background-color: #273746;
                color: #ECF0F1;
            }
        """)
        
        # Create tabs
        self.general_tab = QWidget()
        self.appearance_tab = QWidget()
        self.neural_tab = QWidget()
        self.llm_tab = QWidget()
        self.advanced_tab = QWidget()
        
        # Setup each tab
        self.setup_general_tab()
        self.setup_appearance_tab()
        self.setup_neural_tab()
        self.setup_llm_tab()
        self.setup_advanced_tab()
        
        # Add tabs to widget
        self.tabs.addTab(self.general_tab, "General")
        self.tabs.addTab(self.appearance_tab, "Appearance")
        self.tabs.addTab(self.neural_tab, "Neural Network")
        self.tabs.addTab(self.llm_tab, "LLM Integration")
        self.tabs.addTab(self.advanced_tab, "Advanced")
        
        main_layout.addWidget(self.tabs)
        
        # Save & Reset buttons
        button_container = QHBoxLayout()
        
        # Reset button
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #7F8C8D;
                color: white;
                border-radius: 5px;
                padding: 10px 15px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #95A5A6;
            }
            QPushButton:pressed {
                background-color: #34495E;
            }
        """)
        self.reset_button.clicked.connect(self.reset_settings)
        button_container.addWidget(self.reset_button)
        
        # Spacer
        button_container.addStretch()
        
        # Save button
        self.save_button = QPushButton("Save Settings")
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #2980B9;
                color: white;
                border-radius: 5px;
                padding: 10px 15px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3498DB;
            }
            QPushButton:pressed {
                background-color: #1C587F;
            }
        """)
        self.save_button.clicked.connect(self.save_settings)
        button_container.addWidget(self.save_button)
        
        main_layout.addLayout(button_container)
        
        # Add spacer at the bottom
        main_layout.addStretch()
        
        self.setLayout(main_layout)
        
    def setup_general_tab(self):
        """Setup the general settings tab"""
        layout = QVBoxLayout(self.general_tab)
        layout.setSpacing(15)
        
        # Startup section
        startup_group = QGroupBox("Startup")
        startup_group.setStyleSheet(self.get_group_style())
        startup_layout = QVBoxLayout(startup_group)
        
        # Startup mode
        startup_container = QHBoxLayout()
        startup_label = QLabel("Startup Mode:")
        startup_label.setStyleSheet(self.get_label_style())
        self.startup_combo = QComboBox()
        self.startup_combo.addItems(["Normal", "Minimized", "Fullscreen"])
        self.startup_combo.setStyleSheet(self.get_combo_style())
        startup_container.addWidget(startup_label, 1)
        startup_container.addWidget(self.startup_combo, 2)
        startup_layout.addLayout(startup_container)
        
        # Auto-start checkbox
        auto_start = QCheckBox("Start on system boot")
        auto_start.setStyleSheet(self.get_checkbox_style())
        startup_layout.addWidget(auto_start)
        
        # Auto-update checkbox
        auto_update = QCheckBox("Check for updates on startup")
        auto_update.setChecked(True)
        auto_update.setStyleSheet(self.get_checkbox_style())
        startup_layout.addWidget(auto_update)
        
        layout.addWidget(startup_group)
        
        # File storage section
        files_group = QGroupBox("File Storage")
        files_group.setStyleSheet(self.get_group_style())
        files_layout = QVBoxLayout(files_group)
        
        # Data directory
        data_dir_container = QHBoxLayout()
        data_dir_label = QLabel("Data Directory:")
        data_dir_label.setStyleSheet(self.get_label_style())
        self.data_dir_path = QLabel("C:/Users/jtran/neural_network_project/data")
        self.data_dir_path.setStyleSheet("""
            color: #ECF0F1;
            background-color: #1E2C3A;
            padding: 5px;
            border-radius: 3px;
        """)
        self.browse_button = QPushButton("Browse...")
        self.browse_button.setStyleSheet("""
            QPushButton {
                background-color: #2C3E50;
                color: white;
                border-radius: 3px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #34495E;
            }
        """)
        data_dir_container.addWidget(data_dir_label, 1)
        data_dir_container.addWidget(self.data_dir_path, 2)
        data_dir_container.addWidget(self.browse_button)
        files_layout.addLayout(data_dir_container)
        
        # Auto-save frequency
        autosave_container = QHBoxLayout()
        autosave_label = QLabel("Auto-save frequency (minutes):")
        autosave_label.setStyleSheet(self.get_label_style())
        self.autosave_spinner = QSpinBox()
        self.autosave_spinner.setRange(1, 60)
        self.autosave_spinner.setValue(10)
        self.autosave_spinner.setStyleSheet("""
            QSpinBox {
                background-color: #1E2C3A;
                color: #ECF0F1;
                border-radius: 3px;
                padding: 5px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #2C3E50;
                width: 16px;
                border-radius: 2px;
            }
        """)
        autosave_container.addWidget(autosave_label, 1)
        autosave_container.addWidget(self.autosave_spinner, 2)
        files_layout.addLayout(autosave_container)
        
        layout.addWidget(files_group)
        
        # Memory Management
        memory_group = QGroupBox("Memory Management")
        memory_group.setStyleSheet(self.get_group_style())
        memory_layout = QVBoxLayout(memory_group)
        
        # Max memory entries
        memory_container = QHBoxLayout()
        memory_label = QLabel("Maximum memory entries:")
        memory_label.setStyleSheet(self.get_label_style())
        self.memory_spinner = QSpinBox()
        self.memory_spinner.setRange(100, 10000)
        self.memory_spinner.setSingleStep(100)
        self.memory_spinner.setValue(1000)
        self.memory_spinner.setStyleSheet("""
            QSpinBox {
                background-color: #1E2C3A;
                color: #ECF0F1;
                border-radius: 3px;
                padding: 5px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #2C3E50;
                width: 16px;
                border-radius: 2px;
            }
        """)
        memory_container.addWidget(memory_label, 1)
        memory_container.addWidget(self.memory_spinner, 2)
        memory_layout.addLayout(memory_container)
        
        layout.addWidget(memory_group)
        
    def setup_appearance_tab(self):
        """Setup the appearance settings tab"""
        layout = QVBoxLayout(self.appearance_tab)
        layout.setSpacing(15)
        
        # Theme section
        theme_group = QGroupBox("Theme")
        theme_group.setStyleSheet(self.get_group_style())
        theme_layout = QVBoxLayout(theme_group)
        
        # Theme selection
        theme_container = QHBoxLayout()
        theme_label = QLabel("Color Theme:")
        theme_label.setStyleSheet(self.get_label_style())
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light", "Midnight", "Ocean", "Forest"])
        self.theme_combo.setStyleSheet(self.get_combo_style())
        theme_container.addWidget(theme_label, 1)
        theme_container.addWidget(self.theme_combo, 2)
        theme_layout.addLayout(theme_container)
        
        # Accent color
        accent_container = QHBoxLayout()
        accent_label = QLabel("Accent Color:")
        accent_label.setStyleSheet(self.get_label_style())
        self.accent_combo = QComboBox()
        self.accent_combo.addItems(["Blue", "Green", "Purple", "Orange", "Teal"])
        self.accent_combo.setStyleSheet(self.get_combo_style())
        accent_container.addWidget(accent_label, 1)
        accent_container.addWidget(self.accent_combo, 2)
        theme_layout.addLayout(accent_container)
        
        layout.addWidget(theme_group)
        
        # Font section
        font_group = QGroupBox("Font")
        font_group.setStyleSheet(self.get_group_style())
        font_layout = QVBoxLayout(font_group)
        
        # Font family
        font_container = QHBoxLayout()
        font_label = QLabel("Font Family:")
        font_label.setStyleSheet(self.get_label_style())
        self.font_combo = QComboBox()
        self.font_combo.addItems(["Segoe UI", "Arial", "Roboto", "Open Sans", "Consolas"])
        self.font_combo.setStyleSheet(self.get_combo_style())
        font_container.addWidget(font_label, 1)
        font_container.addWidget(self.font_combo, 2)
        font_layout.addLayout(font_container)
        
        # Font size
        font_size_container = QHBoxLayout()
        font_size_label = QLabel("Font Size:")
        font_size_label.setStyleSheet(self.get_label_style())
        self.font_size_slider = QSlider(Qt.Horizontal)
        self.font_size_slider.setMinimum(8)
        self.font_size_slider.setMaximum(18)
        self.font_size_slider.setValue(12)
        self.font_size_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: #2C3E50;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498DB;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        self.font_size_value = QLabel("12px")
        self.font_size_value.setStyleSheet(self.get_label_style())
        self.font_size_slider.valueChanged.connect(self.update_font_size_value)
        font_size_container.addWidget(font_size_label, 1)
        font_size_container.addWidget(self.font_size_slider, 2)
        font_size_container.addWidget(self.font_size_value)
        font_layout.addLayout(font_size_container)
        
        layout.addWidget(font_group)
        
        # Layout section
        layout_group = QGroupBox("Layout")
        layout_group.setStyleSheet(self.get_group_style())
        layout_inner = QVBoxLayout(layout_group)
        
        # Compact mode
        compact_mode = QCheckBox("Compact Mode")
        compact_mode.setStyleSheet(self.get_checkbox_style())
        layout_inner.addWidget(compact_mode)
        
        # Show tooltips
        show_tooltips = QCheckBox("Show Tooltips")
        show_tooltips.setChecked(True)
        show_tooltips.setStyleSheet(self.get_checkbox_style())
        layout_inner.addWidget(show_tooltips)
        
        # Animations
        enable_animations = QCheckBox("Enable Animations")
        enable_animations.setChecked(True)
        enable_animations.setStyleSheet(self.get_checkbox_style())
        layout_inner.addWidget(enable_animations)
        
        layout.addWidget(layout_group)
        
    def setup_neural_tab(self):
        """Setup the neural network settings tab"""
        layout = QVBoxLayout(self.neural_tab)
        layout.setSpacing(15)
        
        # Model parameters section
        model_group = QGroupBox("Model Parameters")
        model_group.setStyleSheet(self.get_group_style())
        model_layout = QVBoxLayout(model_group)
        
        # Training batch size
        batch_container = QHBoxLayout()
        batch_label = QLabel("Batch Size:")
        batch_label.setStyleSheet(self.get_label_style())
        self.batch_combo = QComboBox()
        self.batch_combo.addItems(["8", "16", "32", "64", "128"])
        self.batch_combo.setCurrentText("32")
        self.batch_combo.setStyleSheet(self.get_combo_style())
        batch_container.addWidget(batch_label, 1)
        batch_container.addWidget(self.batch_combo, 2)
        model_layout.addLayout(batch_container)
        
        # Learning rate
        lr_container = QHBoxLayout()
        lr_label = QLabel("Learning Rate:")
        lr_label.setStyleSheet(self.get_label_style())
        self.lr_combo = QComboBox()
        self.lr_combo.addItems(["0.001", "0.01", "0.1"])
        self.lr_combo.setCurrentText("0.01")
        self.lr_combo.setStyleSheet(self.get_combo_style())
        lr_container.addWidget(lr_label, 1)
        lr_container.addWidget(self.lr_combo, 2)
        model_layout.addLayout(lr_container)
        
        # Hidden layers
        layers_container = QHBoxLayout()
        layers_label = QLabel("Hidden Layers:")
        layers_label.setStyleSheet(self.get_label_style())
        self.layers_combo = QComboBox()
        self.layers_combo.addItems(["[64]", "[128, 64]", "[256, 128, 64]", "[512, 256, 128, 64]"])
        self.layers_combo.setCurrentText("[128, 64]")
        self.layers_combo.setStyleSheet(self.get_combo_style())
        layers_container.addWidget(layers_label, 1)
        layers_container.addWidget(self.layers_combo, 2)
        model_layout.addLayout(layers_container)
        
        layout.addWidget(model_group)
        
        # Training options section
        training_group = QGroupBox("Training Options")
        training_group.setStyleSheet(self.get_group_style())
        training_layout = QVBoxLayout(training_group)
        
        # Auto-train
        auto_train = QCheckBox("Auto-train on new data")
        auto_train.setChecked(True)
        auto_train.setStyleSheet(self.get_checkbox_style())
        training_layout.addWidget(auto_train)
        
        # Training threshold
        threshold_container = QHBoxLayout()
        threshold_label = QLabel("Training threshold (new examples):")
        threshold_label.setStyleSheet(self.get_label_style())
        self.threshold_spinner = QSpinBox()
        self.threshold_spinner.setRange(10, 1000)
        self.threshold_spinner.setSingleStep(10)
        self.threshold_spinner.setValue(50)
        self.threshold_spinner.setStyleSheet("""
            QSpinBox {
                background-color: #1E2C3A;
                color: #ECF0F1;
                border-radius: 3px;
                padding: 5px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #2C3E50;
                width: 16px;
                border-radius: 2px;
            }
        """)
        threshold_container.addWidget(threshold_label, 1)
        threshold_container.addWidget(self.threshold_spinner, 2)
        training_layout.addLayout(threshold_container)
        
        # Save intermediate models
        save_intermediate = QCheckBox("Save intermediate models")
        save_intermediate.setChecked(False)
        save_intermediate.setStyleSheet(self.get_checkbox_style())
        training_layout.addWidget(save_intermediate)
        
        layout.addWidget(training_group)
        
        # Visualization section
        vis_group = QGroupBox("Visualization")
        vis_group.setStyleSheet(self.get_group_style())
        vis_layout = QVBoxLayout(vis_group)
        
        # Detail level
        detail_container = QHBoxLayout()
        detail_label = QLabel("Detail Level:")
        detail_label.setStyleSheet(self.get_label_style())
        self.detail_combo = QComboBox()
        self.detail_combo.addItems(["Low", "Medium", "High", "Maximum"])
        self.detail_combo.setCurrentText("Medium")
        self.detail_combo.setStyleSheet(self.get_combo_style())
        detail_container.addWidget(detail_label, 1)
        detail_container.addWidget(self.detail_combo, 2)
        vis_layout.addLayout(detail_container)
        
        # Frame rate
        fps_container = QHBoxLayout()
        fps_label = QLabel("Refresh Rate (FPS):")
        fps_label.setStyleSheet(self.get_label_style())
        self.fps_spinner = QSpinBox()
        self.fps_spinner.setRange(10, 60)
        self.fps_spinner.setSingleStep(5)
        self.fps_spinner.setValue(30)
        self.fps_spinner.setStyleSheet("""
            QSpinBox {
                background-color: #1E2C3A;
                color: #ECF0F1;
                border-radius: 3px;
                padding: 5px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #2C3E50;
                width: 16px;
                border-radius: 2px;
            }
        """)
        fps_container.addWidget(fps_label, 1)
        fps_container.addWidget(self.fps_spinner, 2)
        vis_layout.addLayout(fps_container)
        
        layout.addWidget(vis_group)
        
    def setup_llm_tab(self):
        """Setup the LLM integration settings tab"""
        layout = QVBoxLayout(self.llm_tab)
        layout.setSpacing(15)
        
        # API configuration section
        api_group = QGroupBox("API Configuration")
        api_group.setStyleSheet(self.get_group_style())
        api_layout = QVBoxLayout(api_group)
        
        # Provider selection
        provider_container = QHBoxLayout()
        provider_label = QLabel("Provider:")
        provider_label.setStyleSheet(self.get_label_style())
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["Mistral AI", "Anthropic", "OpenAI", "Local LLM"])
        self.provider_combo.setStyleSheet(self.get_combo_style())
        provider_container.addWidget(provider_label, 1)
        provider_container.addWidget(self.provider_combo, 2)
        api_layout.addLayout(provider_container)
        
        # API key
        api_key_container = QHBoxLayout()
        api_key_label = QLabel("API Key:")
        api_key_label.setStyleSheet(self.get_label_style())
        self.api_key_path = QLabel("*************")
        self.api_key_path.setStyleSheet("""
            color: #ECF0F1;
            background-color: #1E2C3A;
            padding: 5px;
            border-radius: 3px;
        """)
        self.edit_key_button = QPushButton("Edit...")
        self.edit_key_button.setStyleSheet("""
            QPushButton {
                background-color: #2C3E50;
                color: white;
                border-radius: 3px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #34495E;
            }
        """)
        api_key_container.addWidget(api_key_label, 1)
        api_key_container.addWidget(self.api_key_path, 2)
        api_key_container.addWidget(self.edit_key_button)
        api_layout.addLayout(api_key_container)
        
        layout.addWidget(api_group)
        
        # Model parameters section
        model_group = QGroupBox("Model Parameters")
        model_group.setStyleSheet(self.get_group_style())
        model_layout = QVBoxLayout(model_group)
        
        # Model selection
        model_container = QHBoxLayout()
        model_label = QLabel("Model:")
        model_label.setStyleSheet(self.get_label_style())
        self.model_combo = QComboBox()
        self.model_combo.addItems(["mistral-medium", "mistral-small", "mistral-large", "claude-3-sonnet", "claude-3-haiku", "gpt-4"])
        self.model_combo.setStyleSheet(self.get_combo_style())
        model_container.addWidget(model_label, 1)
        model_container.addWidget(self.model_combo, 2)
        model_layout.addLayout(model_container)
        
        # Temperature setting
        temp_container = QHBoxLayout()
        temp_label = QLabel("Temperature:")
        temp_label.setStyleSheet(self.get_label_style())
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setMinimum(0)
        self.temp_slider.setMaximum(100)
        self.temp_slider.setValue(70)
        self.temp_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: #2C3E50;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498DB;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        self.temp_value = QLabel("0.7")
        self.temp_value.setStyleSheet(self.get_label_style())
        self.temp_slider.valueChanged.connect(self.update_temp_value)
        temp_container.addWidget(temp_label, 1)
        temp_container.addWidget(self.temp_slider, 2)
        temp_container.addWidget(self.temp_value)
        model_layout.addLayout(temp_container)
        
        # Max tokens
        tokens_container = QHBoxLayout()
        tokens_label = QLabel("Max Tokens:")
        tokens_label.setStyleSheet(self.get_label_style())
        self.tokens_spinner = QSpinBox()
        self.tokens_spinner.setRange(10, 4096)
        self.tokens_spinner.setSingleStep(10)
        self.tokens_spinner.setValue(1024)
        self.tokens_spinner.setStyleSheet("""
            QSpinBox {
                background-color: #1E2C3A;
                color: #ECF0F1;
                border-radius: 3px;
                padding: 5px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #2C3E50;
                width: 16px;
                border-radius: 2px;
            }
        """)
        tokens_container.addWidget(tokens_label, 1)
        tokens_container.addWidget(self.tokens_spinner, 2)
        model_layout.addLayout(tokens_container)
        
        layout.addWidget(model_group)
        
        # Integration options
        integration_group = QGroupBox("Integration Options")
        integration_group.setStyleSheet(self.get_group_style())
        integration_layout = QVBoxLayout(integration_group)
        
        # Default weight
        weight_container = QHBoxLayout()
        weight_label = QLabel("Default LLM Weight:")
        weight_label.setStyleSheet(self.get_label_style())
        self.weight_slider = QSlider(Qt.Horizontal)
        self.weight_slider.setMinimum(0)
        self.weight_slider.setMaximum(100)
        self.weight_slider.setValue(50)
        self.weight_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: #2C3E50;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498DB;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        self.weight_value = QLabel("0.5")
        self.weight_value.setStyleSheet(self.get_label_style())
        self.weight_slider.valueChanged.connect(self.update_weight_value)
        weight_container.addWidget(weight_label, 1)
        weight_container.addWidget(self.weight_slider, 2)
        weight_container.addWidget(self.weight_value)
        integration_layout.addLayout(weight_container)
        
        # Dynamic weighting
        dynamic_weighting = QCheckBox("Dynamic LLM Weighting")
        dynamic_weighting.setChecked(True)
        dynamic_weighting.setStyleSheet(self.get_checkbox_style())
        integration_layout.addWidget(dynamic_weighting)
        
        layout.addWidget(integration_group)
        
    def setup_advanced_tab(self):
        """Setup the advanced settings tab"""
        layout = QVBoxLayout(self.advanced_tab)
        layout.setSpacing(15)
        
        # Performance section
        perf_group = QGroupBox("Performance")
        perf_group.setStyleSheet(self.get_group_style())
        perf_layout = QVBoxLayout(perf_group)
        
        # Multi-threading
        threading_container = QHBoxLayout()
        threading_label = QLabel("Processing Threads:")
        threading_label.setStyleSheet(self.get_label_style())
        self.threads_spinner = QSpinBox()
        self.threads_spinner.setRange(1, 16)
        self.threads_spinner.setValue(4)
        self.threads_spinner.setStyleSheet("""
            QSpinBox {
                background-color: #1E2C3A;
                color: #ECF0F1;
                border-radius: 3px;
                padding: 5px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #2C3E50;
                width: 16px;
                border-radius: 2px;
            }
        """)
        threading_container.addWidget(threading_label, 1)
        threading_container.addWidget(self.threads_spinner, 2)
        perf_layout.addLayout(threading_container)
        
        # GPU acceleration
        gpu_accel = QCheckBox("Enable GPU Acceleration")
        gpu_accel.setChecked(True)
        gpu_accel.setStyleSheet(self.get_checkbox_style())
        perf_layout.addWidget(gpu_accel)
        
        # Memory limit
        memory_container = QHBoxLayout()
        memory_label = QLabel("Memory Limit (MB):")
        memory_label.setStyleSheet(self.get_label_style())
        self.memory_spinner = QSpinBox()
        self.memory_spinner.setRange(512, 8192)
        self.memory_spinner.setSingleStep(512)
        self.memory_spinner.setValue(2048)
        self.memory_spinner.setStyleSheet("""
            QSpinBox {
                background-color: #1E2C3A;
                color: #ECF0F1;
                border-radius: 3px;
                padding: 5px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #2C3E50;
                width: 16px;
                border-radius: 2px;
            }
        """)
        memory_container.addWidget(memory_label, 1)
        memory_container.addWidget(self.memory_spinner, 2)
        perf_layout.addLayout(memory_container)
        
        layout.addWidget(perf_group)
        
        # Debugging section
        debug_group = QGroupBox("Debugging")
        debug_group.setStyleSheet(self.get_group_style())
        debug_layout = QVBoxLayout(debug_group)
        
        # Debug mode
        debug_mode = QCheckBox("Enable Debug Mode")
        debug_mode.setStyleSheet(self.get_checkbox_style())
        debug_layout.addWidget(debug_mode)
        
        # Log level
        log_container = QHBoxLayout()
        log_label = QLabel("Log Level:")
        log_label.setStyleSheet(self.get_label_style())
        self.log_combo = QComboBox()
        self.log_combo.addItems(["ERROR", "WARNING", "INFO", "DEBUG"])
        self.log_combo.setCurrentText("INFO")
        self.log_combo.setStyleSheet(self.get_combo_style())
        log_container.addWidget(log_label, 1)
        log_container.addWidget(self.log_combo, 2)
        debug_layout.addLayout(log_container)
        
        # Verbose console
        verbose_console = QCheckBox("Verbose Console Output")
        verbose_console.setStyleSheet(self.get_checkbox_style())
        debug_layout.addWidget(verbose_console)
        
        layout.addWidget(debug_group)
        
        # Experimental Features
        exp_group = QGroupBox("Experimental Features")
        exp_group.setStyleSheet(self.get_group_style())
        exp_layout = QVBoxLayout(exp_group)
        
        # Beta features
        beta_features = QCheckBox("Enable Beta Features")
        beta_features.setStyleSheet(self.get_checkbox_style())
        exp_layout.addWidget(beta_features)
        
        # Experimental algorithm
        experimental_algo = QCheckBox("Use Experimental Processing Algorithm")
        experimental_algo.setStyleSheet(self.get_checkbox_style())
        exp_layout.addWidget(experimental_algo)
        
        layout.addWidget(exp_group)
    
    def update_font_size_value(self, value):
        """Update the font size value label when slider changes"""
        self.font_size_value.setText(f"{value}px")
        
    def update_temp_value(self, value):
        """Update the temperature value label when slider changes"""
        self.temp_value.setText(f"{value / 100:.1f}")
        
    def update_weight_value(self, value):
        """Update the weight value label when slider changes"""
        self.weight_value.setText(f"{value / 100:.1f}")
        
    def get_label_style(self):
        """Get the standard label style"""
        return "color: #ECF0F1; font-size: 14px;"
        
    def get_combo_style(self):
        """Get the standard combo box style"""
        return """
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 5px;
            padding: 5px;
            font-size: 14px;
        """
        
    def get_checkbox_style(self):
        """Get the standard checkbox style"""
        return """
            QCheckBox {
                color: #ECF0F1;
                font-size: 14px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 4px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #2C3E50;
                border: 2px solid #34495E;
            }
            QCheckBox::indicator:checked {
                background-color: #3498DB;
                border: 2px solid #2980B9;
            }
        """
        
    def get_group_style(self):
        """Get the standard group box style"""
        return """
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                border: 1px solid #2C3E50;
                border-radius: 5px;
                margin-top: 10px;
                padding: 5px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #3498DB;
            }
        """
        
    def reset_settings(self):
        """Reset all settings to defaults"""
        # Here you would reset all widgets to their default values
        pass
        
    def save_settings(self):
        """Collect and emit all settings"""
        # Here you would collect all settings from the widgets
        settings = {
            "general": {
                "startup_mode": self.startup_combo.currentText(),
                # Add other general settings
            },
            "appearance": {
                "theme": self.theme_combo.currentText(),
                "font_size": self.font_size_slider.value(),
                # Add other appearance settings
            },
            "neural": {
                "batch_size": int(self.batch_combo.currentText()),
                "learning_rate": float(self.lr_combo.currentText()),
                # Add other neural settings
            },
            "llm": {
                "provider": self.provider_combo.currentText(),
                "model": self.model_combo.currentText(),
                "temperature": float(self.temp_value.text()),
                # Add other LLM settings
            },
            "advanced": {
                "threads": self.threads_spinner.value(),
                "log_level": self.log_combo.currentText(),
                # Add other advanced settings
            }
        }
        
        self.settings_saved.emit(settings) 