from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                          QPushButton, QProgressBar, QComboBox, QCheckBox,
                          QFrame, QSplitter, QGroupBox, QGridLayout, QSpinBox,
                          QDoubleSpinBox, QFileDialog, QTabWidget, QTableWidget,
                          QTableWidgetItem, QHeaderView, QScrollArea, QLineEdit)
from PyQt5.QtGui import QIcon, QFont, QColor, QPen, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QSize

class TrainingMetricsWidget(QWidget):
    """Widget for displaying training metrics and charts"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        """Initialize the training metrics UI"""
        main_layout = QVBoxLayout(self)
        
        # Metrics tabs
        metrics_tabs = QTabWidget()
        metrics_tabs.setStyleSheet("""
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
        
        # Loss chart tab
        loss_tab = QWidget()
        loss_layout = QVBoxLayout(loss_tab)
        
        # Placeholder for loss chart - in a real implementation, this would use matplotlib or another charting library
        loss_chart = QLabel("Loss Chart Placeholder")
        loss_chart.setAlignment(Qt.AlignCenter)
        loss_chart.setStyleSheet("""
            background-color: #1E2C3A;
            color: #3498DB;
            border: 1px dashed #3498DB;
            border-radius: 5px;
            padding: 40px;
            font-size: 16px;
        """)
        loss_chart.setMinimumHeight(200)
        loss_layout.addWidget(loss_chart)
        
        # Accuracy chart tab
        accuracy_tab = QWidget()
        accuracy_layout = QVBoxLayout(accuracy_tab)
        
        # Placeholder for accuracy chart
        accuracy_chart = QLabel("Accuracy Chart Placeholder")
        accuracy_chart.setAlignment(Qt.AlignCenter)
        accuracy_chart.setStyleSheet("""
            background-color: #1E2C3A;
            color: #3498DB;
            border: 1px dashed #3498DB;
            border-radius: 5px;
            padding: 40px;
            font-size: 16px;
        """)
        accuracy_chart.setMinimumHeight(200)
        accuracy_layout.addWidget(accuracy_chart)
        
        # Gradient tab
        gradient_tab = QWidget()
        gradient_layout = QVBoxLayout(gradient_tab)
        
        # Placeholder for gradient chart
        gradient_chart = QLabel("Gradient Norms Chart Placeholder")
        gradient_chart.setAlignment(Qt.AlignCenter)
        gradient_chart.setStyleSheet("""
            background-color: #1E2C3A;
            color: #3498DB;
            border: 1px dashed #3498DB;
            border-radius: 5px;
            padding: 40px;
            font-size: 16px;
        """)
        gradient_chart.setMinimumHeight(200)
        gradient_layout.addWidget(gradient_chart)
        
        # Add tabs to the tab widget
        metrics_tabs.addTab(loss_tab, "Loss")
        metrics_tabs.addTab(accuracy_tab, "Accuracy")
        metrics_tabs.addTab(gradient_tab, "Gradients")
        
        main_layout.addWidget(metrics_tabs)
        
        # Current metrics table
        metrics_group = QGroupBox("Current Metrics")
        metrics_group.setStyleSheet("""
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
        """)
        
        metrics_layout = QGridLayout(metrics_group)
        metrics_layout.setColumnStretch(1, 1)
        metrics_layout.setColumnStretch(3, 1)
        
        # Training loss
        metrics_layout.addWidget(QLabel("Training Loss:"), 0, 0)
        self.train_loss = QLabel("0.0000")
        self.train_loss.setStyleSheet("color: #ECF0F1; font-weight: bold;")
        metrics_layout.addWidget(self.train_loss, 0, 1)
        
        # Validation loss
        metrics_layout.addWidget(QLabel("Validation Loss:"), 0, 2)
        self.val_loss = QLabel("0.0000")
        self.val_loss.setStyleSheet("color: #ECF0F1; font-weight: bold;")
        metrics_layout.addWidget(self.val_loss, 0, 3)
        
        # Training accuracy
        metrics_layout.addWidget(QLabel("Training Accuracy:"), 1, 0)
        self.train_acc = QLabel("0.00%")
        self.train_acc.setStyleSheet("color: #ECF0F1; font-weight: bold;")
        metrics_layout.addWidget(self.train_acc, 1, 1)
        
        # Validation accuracy
        metrics_layout.addWidget(QLabel("Validation Accuracy:"), 1, 2)
        self.val_acc = QLabel("0.00%")
        self.val_acc.setStyleSheet("color: #ECF0F1; font-weight: bold;")
        metrics_layout.addWidget(self.val_acc, 1, 3)
        
        # Learning rate
        metrics_layout.addWidget(QLabel("Learning Rate:"), 2, 0)
        self.learning_rate = QLabel("0.001")
        self.learning_rate.setStyleSheet("color: #ECF0F1; font-weight: bold;")
        metrics_layout.addWidget(self.learning_rate, 2, 1)
        
        # Epoch
        metrics_layout.addWidget(QLabel("Epoch:"), 2, 2)
        self.epoch = QLabel("0/100")
        self.epoch.setStyleSheet("color: #ECF0F1; font-weight: bold;")
        metrics_layout.addWidget(self.epoch, 2, 3)
        
        main_layout.addWidget(metrics_group)
        
        # Initialize with mock data
        self.update_mock_metrics()
        
    def update_mock_metrics(self):
        """Update metrics with mock data"""
        import random
        
        # Simulate decreasing loss
        epoch_num = random.randint(1, 100)
        max_epochs = 100
        
        train_loss = 2.0 / (1 + 0.1 * epoch_num) + random.uniform(-0.05, 0.05)
        val_loss = train_loss + random.uniform(0.05, 0.2)
        
        train_acc = min(95, 100 - 100 / (1 + 0.05 * epoch_num)) + random.uniform(-2, 2)
        val_acc = train_acc - random.uniform(5, 15)
        
        # Update labels
        self.train_loss.setText(f"{train_loss:.4f}")
        self.val_loss.setText(f"{val_loss:.4f}")
        self.train_acc.setText(f"{train_acc:.2f}%")
        self.val_acc.setText(f"{val_acc:.2f}%")
        self.learning_rate.setText(f"{0.001 / (1 + 0.01 * epoch_num):.6f}")
        self.epoch.setText(f"{epoch_num}/{max_epochs}")

class ModelConfigWidget(QWidget):
    """Widget for configuring neural network model parameters"""
    
    config_changed = pyqtSignal(dict)  # Signal emitted when configuration changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        """Initialize the model configuration UI"""
        main_layout = QVBoxLayout(self)
        
        # Architecture configuration
        arch_group = QGroupBox("Network Architecture")
        arch_group.setStyleSheet("""
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
        """)
        
        arch_layout = QGridLayout(arch_group)
        
        # Architecture type
        arch_layout.addWidget(QLabel("Architecture:"), 0, 0)
        self.arch_combo = QComboBox()
        self.arch_combo.addItems(["Feedforward", "CNN", "RNN", "LSTM", "Transformer"])
        self.arch_combo.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 5px;
            padding: 5px;
        """)
        arch_layout.addWidget(self.arch_combo, 0, 1)
        
        # Hidden layers
        arch_layout.addWidget(QLabel("Hidden Layers:"), 1, 0)
        hidden_layout = QHBoxLayout()
        
        self.layers_spin = QSpinBox()
        self.layers_spin.setRange(1, 10)
        self.layers_spin.setValue(2)
        self.layers_spin.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 3px;
            padding: 5px;
        """)
        hidden_layout.addWidget(self.layers_spin)
        
        # Layer size presets
        self.layer_size_combo = QComboBox()
        self.layer_size_combo.addItems(["Small [32, 16]", "Medium [128, 64]", "Large [512, 256, 128]", "Custom"])
        self.layer_size_combo.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 5px;
            padding: 5px;
        """)
        hidden_layout.addWidget(self.layer_size_combo)
        
        arch_layout.addLayout(hidden_layout, 1, 1)
        
        # Custom layer sizes (initially hidden)
        arch_layout.addWidget(QLabel("Custom Sizes:"), 2, 0)
        self.custom_layers = QLineEdit("128, 64")
        self.custom_layers.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 5px;
            padding: 5px;
        """)
        self.custom_layers.setPlaceholderText("e.g., 128, 64, 32")
        arch_layout.addWidget(self.custom_layers, 2, 1)
        
        # Activation function
        arch_layout.addWidget(QLabel("Activation:"), 3, 0)
        self.activation_combo = QComboBox()
        self.activation_combo.addItems(["ReLU", "Sigmoid", "Tanh", "LeakyReLU", "ELU"])
        self.activation_combo.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 5px;
            padding: 5px;
        """)
        arch_layout.addWidget(self.activation_combo, 3, 1)
        
        main_layout.addWidget(arch_group)
        
        # Training parameters
        train_group = QGroupBox("Training Parameters")
        train_group.setStyleSheet("""
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
        """)
        
        train_layout = QGridLayout(train_group)
        
        # Batch size
        train_layout.addWidget(QLabel("Batch Size:"), 0, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setValue(32)
        self.batch_spin.setSingleStep(8)
        self.batch_spin.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 3px;
            padding: 5px;
        """)
        train_layout.addWidget(self.batch_spin, 0, 1)
        
        # Epochs
        train_layout.addWidget(QLabel("Epochs:"), 1, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        self.epochs_spin.setSingleStep(10)
        self.epochs_spin.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 3px;
            padding: 5px;
        """)
        train_layout.addWidget(self.epochs_spin, 1, 1)
        
        # Learning rate
        train_layout.addWidget(QLabel("Learning Rate:"), 2, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 3px;
            padding: 5px;
        """)
        train_layout.addWidget(self.lr_spin, 2, 1)
        
        # Optimizer
        train_layout.addWidget(QLabel("Optimizer:"), 3, 0)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "SGD", "RMSprop", "Adagrad", "Adadelta"])
        self.optimizer_combo.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 5px;
            padding: 5px;
        """)
        train_layout.addWidget(self.optimizer_combo, 3, 1)
        
        # Regularization
        train_layout.addWidget(QLabel("Regularization:"), 4, 0)
        reg_layout = QHBoxLayout()
        
        self.l1_check = QCheckBox("L1")
        self.l1_check.setStyleSheet("""
            QCheckBox {
                color: #ECF0F1;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #1E2C3A;
                border: 1px solid #2C3E50;
            }
            QCheckBox::indicator:checked {
                background-color: #3498DB;
                border: 1px solid #2980B9;
            }
        """)
        reg_layout.addWidget(self.l1_check)
        
        self.l2_check = QCheckBox("L2")
        self.l2_check.setChecked(True)
        self.l2_check.setStyleSheet("""
            QCheckBox {
                color: #ECF0F1;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #1E2C3A;
                border: 1px solid #2C3E50;
            }
            QCheckBox::indicator:checked {
                background-color: #3498DB;
                border: 1px solid #2980B9;
            }
        """)
        reg_layout.addWidget(self.l2_check)
        
        self.dropout_check = QCheckBox("Dropout")
        self.dropout_check.setChecked(True)
        self.dropout_check.setStyleSheet("""
            QCheckBox {
                color: #ECF0F1;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #1E2C3A;
                border: 1px solid #2C3E50;
            }
            QCheckBox::indicator:checked {
                background-color: #3498DB;
                border: 1px solid #2980B9;
            }
        """)
        reg_layout.addWidget(self.dropout_check)
        
        train_layout.addLayout(reg_layout, 4, 1)
        
        # Dropout rate
        train_layout.addWidget(QLabel("Dropout Rate:"), 5, 0)
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.9)
        self.dropout_spin.setValue(0.2)
        self.dropout_spin.setDecimals(2)
        self.dropout_spin.setSingleStep(0.05)
        self.dropout_spin.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 3px;
            padding: 5px;
        """)
        train_layout.addWidget(self.dropout_spin, 5, 1)
        
        main_layout.addWidget(train_group)
        
        # Data group
        data_group = QGroupBox("Data Configuration")
        data_group.setStyleSheet("""
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
        """)
        
        data_layout = QGridLayout(data_group)
        
        # Data source
        data_layout.addWidget(QLabel("Data Source:"), 0, 0)
        self.data_combo = QComboBox()
        self.data_combo.addItems(["Memory Database", "External Dataset", "Synthetic Data"])
        self.data_combo.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 5px;
            padding: 5px;
        """)
        data_layout.addWidget(self.data_combo, 0, 1)
        
        # Train/Val split
        data_layout.addWidget(QLabel("Train/Val Split:"), 1, 0)
        self.split_spin = QDoubleSpinBox()
        self.split_spin.setRange(0.5, 0.95)
        self.split_spin.setValue(0.8)
        self.split_spin.setDecimals(2)
        self.split_spin.setSingleStep(0.05)
        self.split_spin.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 3px;
            padding: 5px;
        """)
        data_layout.addWidget(self.split_spin, 1, 1)
        
        # Augmentation
        data_layout.addWidget(QLabel("Augmentation:"), 2, 0)
        self.augment_check = QCheckBox("Enable Data Augmentation")
        self.augment_check.setStyleSheet("""
            QCheckBox {
                color: #ECF0F1;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #1E2C3A;
                border: 1px solid #2C3E50;
            }
            QCheckBox::indicator:checked {
                background-color: #3498DB;
                border: 1px solid #2980B9;
            }
        """)
        data_layout.addWidget(self.augment_check, 2, 1)
        
        main_layout.addWidget(data_group)
        
        # Add a spacer to push everything up
        main_layout.addStretch()
        
        # Connect signals
        self.layer_size_combo.currentTextChanged.connect(self.update_layer_size_fields)
        
    def update_layer_size_fields(self, text):
        """Update custom layer size field based on selection"""
        if text == "Custom":
            self.custom_layers.setEnabled(True)
        else:
            self.custom_layers.setEnabled(False)
            if text == "Small [32, 16]":
                self.custom_layers.setText("32, 16")
            elif text == "Medium [128, 64]":
                self.custom_layers.setText("128, 64")
            elif text == "Large [512, 256, 128]":
                self.custom_layers.setText("512, 256, 128")
        
    def get_config(self):
        """Get the current configuration as a dictionary"""
        config = {
            "architecture": {
                "type": self.arch_combo.currentText(),
                "hidden_layers": self.layers_spin.value(),
                "layer_sizes": self.custom_layers.text().split(","),
                "activation": self.activation_combo.currentText()
            },
            "training": {
                "batch_size": self.batch_spin.value(),
                "epochs": self.epochs_spin.value(),
                "learning_rate": self.lr_spin.value(),
                "optimizer": self.optimizer_combo.currentText(),
                "l1_reg": self.l1_check.isChecked(),
                "l2_reg": self.l2_check.isChecked(),
                "dropout": self.dropout_check.isChecked(),
                "dropout_rate": self.dropout_spin.value()
            },
            "data": {
                "source": self.data_combo.currentText(),
                "train_val_split": self.split_spin.value(),
                "augmentation": self.augment_check.isChecked()
            }
        }
        return config

class TrainingPanel(QWidget):
    """Panel for configuring and monitoring neural network training"""
    
    training_started = pyqtSignal()
    training_stopped = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_training = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_training_progress)
        self.initUI()
        
    def initUI(self):
        """Initialize the training panel UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)
        
        # Header area
        header_layout = QHBoxLayout()
        
        # Title
        title = QLabel("Neural Network Training")
        title.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #3498DB;
            margin-bottom: 10px;
        """)
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Model selection
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Main Neural Network", "Memory Encoding Network", "Language Processing Network"])
        self.model_combo.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 5px;
            padding: 8px;
            min-width: 200px;
        """)
        header_layout.addWidget(self.model_combo)
        
        main_layout.addLayout(header_layout)
        
        # Split view for configuration and monitoring
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        
        # Left side - configuration
        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
        
        self.config_widget = ModelConfigWidget()
        config_scroll.setWidget(self.config_widget)
        
        # Right side - metrics and monitoring
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)
        
        # Status and controls
        status_group = QGroupBox("Training Status")
        status_group.setStyleSheet("""
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
        """)
        
        status_layout = QVBoxLayout(status_group)
        
        # Status label
        self.status_label = QLabel("Ready to Train")
        self.status_label.setStyleSheet("""
            color: #ECF0F1;
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 10px;
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #2C3E50;
                border-radius: 5px;
                background-color: #1E2C3A;
                color: #ECF0F1;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #3498DB;
                border-radius: 4px;
            }
        """)
        status_layout.addWidget(self.progress_bar)
        
        # Action buttons
        actions_layout = QHBoxLayout()
        
        # Start/Stop button
        self.train_button = QPushButton("Start Training")
        self.train_button.setCheckable(True)
        self.train_button.setStyleSheet("""
            QPushButton {
                background-color: #27AE60;
                color: white;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2ECC71;
            }
            QPushButton:checked {
                background-color: #E74C3C;
            }
            QPushButton:checked:hover {
                background-color: #F39C12;
            }
        """)
        self.train_button.clicked.connect(self.toggle_training)
        actions_layout.addWidget(self.train_button)
        
        # Save model button
        self.save_button = QPushButton("Save Model")
        self.save_button.setEnabled(False)
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #2980B9;
                color: white;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3498DB;
            }
            QPushButton:disabled {
                background-color: #7F8C8D;
                color: #BDC3C7;
            }
        """)
        self.save_button.clicked.connect(self.save_model)
        actions_layout.addWidget(self.save_button)
        
        status_layout.addLayout(actions_layout)
        
        metrics_layout.addWidget(status_group)
        
        # Training metrics
        self.metrics_widget = TrainingMetricsWidget()
        metrics_layout.addWidget(self.metrics_widget)
        
        # Add widgets to splitter
        splitter.addWidget(config_scroll)
        splitter.addWidget(metrics_widget)
        
        # Set initial sizes (40% config, 60% metrics)
        splitter.setSizes([400, 600])
        
        main_layout.addWidget(splitter)
        
    def toggle_training(self, checked):
        """Start or stop training based on button state"""
        if checked:
            self.start_training()
        else:
            self.stop_training()
            
    def start_training(self):
        """Start the training process"""
        self.is_training = True
        self.train_button.setText("Stop Training")
        self.status_label.setText("Training in Progress...")
        self.progress_bar.setValue(0)
        
        # Disable configuration while training
        self.config_widget.setEnabled(False)
        
        # Start timer to simulate training progress
        self.timer.start(1000)  # Update every second
        
        # Emit signal
        self.training_started.emit()
        
    def stop_training(self):
        """Stop the training process"""
        self.is_training = False
        self.train_button.setChecked(False)
        self.train_button.setText("Start Training")
        
        if self.progress_bar.value() >= 100:
            self.status_label.setText("Training Complete")
            self.save_button.setEnabled(True)
        else:
            self.status_label.setText("Training Stopped")
        
        # Stop timer
        self.timer.stop()
        
        # Re-enable configuration
        self.config_widget.setEnabled(True)
        
        # Emit signal
        self.training_stopped.emit()
        
    def update_training_progress(self):
        """Update training progress for demonstration"""
        current = self.progress_bar.value()
        
        if current < 100:
            # Increment progress
            self.progress_bar.setValue(current + 1)
            
            # Update metrics display
            self.metrics_widget.update_mock_metrics()
        else:
            # Training complete
            self.stop_training()
            self.status_label.setText("Training Complete")
            self.save_button.setEnabled(True)
            
    def save_model(self):
        """Save the trained model"""
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Neural Network Model", "", "Model Files (*.model *.h5 *.pth);;All Files (*)"
        )
        
        if file_name:
            # Here we would actually save the model in a real implementation
            self.status_label.setText(f"Model Saved: {file_name}")
            # Here would be code to actually save the model 