from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                          QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
                          QHeaderView, QGroupBox, QGridLayout, QLineEdit, QToolButton,
                          QDialog, QFileDialog, QMessageBox, QSplitter, QListWidget,
                          QListWidgetItem, QCheckBox, QSpinBox, QTabWidget, QRadioButton,
                          QProgressBar, QScrollArea)
from PyQt5.QtGui import QIcon, QFont, QColor, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, QSize

class DatasetStatsWidget(QWidget):
    """Widget for displaying dataset statistics and info"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        """Initialize the dataset stats UI"""
        main_layout = QVBoxLayout(self)
        
        # Key metrics grid
        metrics_group = QGroupBox("Dataset Metrics")
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
        
        # Total samples
        metrics_layout.addWidget(QLabel("Total Samples:"), 0, 0)
        self.total_samples = QLabel("1024")
        self.total_samples.setStyleSheet("color: #ECF0F1; font-weight: bold;")
        metrics_layout.addWidget(self.total_samples, 0, 1)
        
        # Input dimensions
        metrics_layout.addWidget(QLabel("Input Dimensions:"), 0, 2)
        self.input_dims = QLabel("28 x 28")
        self.input_dims.setStyleSheet("color: #ECF0F1; font-weight: bold;")
        metrics_layout.addWidget(self.input_dims, 0, 3)
        
        # Classes
        metrics_layout.addWidget(QLabel("Classes/Labels:"), 1, 0)
        self.num_classes = QLabel("10")
        self.num_classes.setStyleSheet("color: #ECF0F1; font-weight: bold;")
        metrics_layout.addWidget(self.num_classes, 1, 1)
        
        # Balance
        metrics_layout.addWidget(QLabel("Class Balance:"), 1, 2)
        self.class_balance = QLabel("Balanced")
        self.class_balance.setStyleSheet("color: #ECF0F1; font-weight: bold;")
        metrics_layout.addWidget(self.class_balance, 1, 3)
        
        # Data type
        metrics_layout.addWidget(QLabel("Data Type:"), 2, 0)
        self.data_type = QLabel("Image")
        self.data_type.setStyleSheet("color: #ECF0F1; font-weight: bold;")
        metrics_layout.addWidget(self.data_type, 2, 1)
        
        # Last updated
        metrics_layout.addWidget(QLabel("Last Updated:"), 2, 2)
        self.last_updated = QLabel("2023-09-15")
        self.last_updated.setStyleSheet("color: #ECF0F1; font-weight: bold;")
        metrics_layout.addWidget(self.last_updated, 2, 3)
        
        main_layout.addWidget(metrics_group)
        
        # Distribution chart (placeholder)
        dist_group = QGroupBox("Class Distribution")
        dist_group.setStyleSheet("""
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
        
        dist_layout = QVBoxLayout(dist_group)
        
        dist_chart = QLabel("Class Distribution Chart Placeholder")
        dist_chart.setAlignment(Qt.AlignCenter)
        dist_chart.setStyleSheet("""
            background-color: #1E2C3A;
            color: #3498DB;
            border: 1px dashed #3498DB;
            border-radius: 5px;
            padding: 40px;
            font-size: 16px;
        """)
        dist_chart.setMinimumHeight(150)
        dist_layout.addWidget(dist_chart)
        
        main_layout.addWidget(dist_group)
        
        # Statistics (placeholder)
        stats_group = QGroupBox("Statistical Summary")
        stats_group.setStyleSheet("""
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
        
        stats_layout = QVBoxLayout(stats_group)
        
        # Table for stats
        stats_table = QTableWidget(5, 4)
        stats_table.setHorizontalHeaderLabels(["Metric", "Min", "Mean", "Max"])
        stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        stats_table.setStyleSheet("""
            QTableWidget {
                background-color: #1E2C3A;
                color: #ECF0F1;
                border: none;
                gridline-color: #2C3E50;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #2C3E50;
                color: #ECF0F1;
                padding: 5px;
                border: 1px solid #34495E;
            }
        """)
        
        # Fill with sample data
        metrics = ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"]
        for i, metric in enumerate(metrics):
            stats_table.setItem(i, 0, QTableWidgetItem(metric))
            stats_table.setItem(i, 1, QTableWidgetItem(f"{0.1 * (i+1):.2f}"))
            stats_table.setItem(i, 2, QTableWidgetItem(f"{0.5 * (i+1):.2f}"))
            stats_table.setItem(i, 3, QTableWidgetItem(f"{0.9 * (i+1):.2f}"))
        
        stats_layout.addWidget(stats_table)
        
        main_layout.addWidget(stats_group)
        main_layout.addStretch()

class DataExplorerWidget(QWidget):
    """Widget for exploring dataset samples"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        """Initialize the data explorer UI"""
        main_layout = QVBoxLayout(self)
        
        # Search and filter
        search_layout = QHBoxLayout()
        
        # Search box
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search for features or values...")
        self.search_box.setStyleSheet("""
            QLineEdit {
                background-color: #1E2C3A;
                color: #ECF0F1;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
                border: 1px solid #2C3E50;
            }
            QLineEdit:focus {
                border: 1px solid #3498DB;
            }
        """)
        search_layout.addWidget(self.search_box)
        
        # Filter by class
        self.class_filter = QComboBox()
        self.class_filter.addItems(["All Classes", "Class 1", "Class 2", "Class 3"])
        self.class_filter.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 5px;
            padding: 5px;
        """)
        search_layout.addWidget(self.class_filter)
        
        # Show only button
        self.show_outliers = QCheckBox("Show Outliers")
        self.show_outliers.setStyleSheet("""
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
        search_layout.addWidget(self.show_outliers)
        
        main_layout.addLayout(search_layout)
        
        # Data table
        self.data_table = QTableWidget(20, 6)
        self.data_table.setHorizontalHeaderLabels(["ID", "Class", "Feature 1", "Feature 2", "Feature 3", "Feature 4"])
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.data_table.setStyleSheet("""
            QTableWidget {
                background-color: #1E2C3A;
                color: #ECF0F1;
                border: none;
                gridline-color: #2C3E50;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #2C3E50;
                color: #ECF0F1;
                padding: 5px;
                border: 1px solid #34495E;
            }
        """)
        
        # Fill with sample data
        for i in range(20):
            # ID
            self.data_table.setItem(i, 0, QTableWidgetItem(f"S{i+1:04d}"))
            
            # Class
            class_idx = i % 3 + 1
            class_item = QTableWidgetItem(f"Class {class_idx}")
            if class_idx == 1:
                class_item.setForeground(QColor(52, 152, 219))  # Blue
            elif class_idx == 2:
                class_item.setForeground(QColor(46, 204, 113))  # Green
            else:
                class_item.setForeground(QColor(155, 89, 182))  # Purple
            self.data_table.setItem(i, 1, class_item)
            
            # Feature values
            for j in range(2, 6):
                value = 0.1 * (i + j) + (j - 2) * 0.05
                self.data_table.setItem(i, j, QTableWidgetItem(f"{value:.2f}"))
        
        main_layout.addWidget(self.data_table)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Page navigation
        page_layout = QHBoxLayout()
        
        self.prev_button = QPushButton("« Previous")
        self.prev_button.setStyleSheet("""
            QPushButton {
                background-color: #2C3E50;
                color: white;
                border-radius: 5px;
                padding: 5px 10px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #34495E;
            }
        """)
        page_layout.addWidget(self.prev_button)
        
        self.page_label = QLabel("Page 1 of 10")
        self.page_label.setStyleSheet("color: #ECF0F1;")
        page_layout.addWidget(self.page_label)
        
        self.next_button = QPushButton("Next »")
        self.next_button.setStyleSheet("""
            QPushButton {
                background-color: #2C3E50;
                color: white;
                border-radius: 5px;
                padding: 5px 10px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #34495E;
            }
        """)
        page_layout.addWidget(self.next_button)
        
        controls_layout.addLayout(page_layout)
        
        controls_layout.addStretch()
        
        # Export button
        self.export_button = QPushButton("Export View")
        self.export_button.setStyleSheet("""
            QPushButton {
                background-color: #2980B9;
                color: white;
                border-radius: 5px;
                padding: 8px 15px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3498DB;
            }
        """)
        controls_layout.addWidget(self.export_button)
        
        main_layout.addLayout(controls_layout)

class DataPrepWidget(QWidget):
    """Widget for data preparation and preprocessing"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        """Initialize the data preparation UI"""
        main_layout = QVBoxLayout(self)
        
        # Input/Output section
        io_group = QGroupBox("Data Source & Destination")
        io_group.setStyleSheet("""
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
        
        io_layout = QGridLayout(io_group)
        
        # Input source
        io_layout.addWidget(QLabel("Input Source:"), 0, 0)
        self.input_source = QComboBox()
        self.input_source.addItems(["Memory Database", "File Import", "Synthetic Generator"])
        self.input_source.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 5px;
            padding: 5px;
        """)
        io_layout.addWidget(self.input_source, 0, 1)
        
        # Source path
        io_layout.addWidget(QLabel("Source Path:"), 1, 0)
        source_path_layout = QHBoxLayout()
        
        self.source_path = QLineEdit()
        self.source_path.setPlaceholderText("/path/to/data")
        self.source_path.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 5px;
            padding: 5px;
        """)
        source_path_layout.addWidget(self.source_path)
        
        self.browse_source = QPushButton("Browse...")
        self.browse_source.setStyleSheet("""
            QPushButton {
                background-color: #2C3E50;
                color: white;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #34495E;
            }
        """)
        source_path_layout.addWidget(self.browse_source)
        
        io_layout.addLayout(source_path_layout, 1, 1)
        
        # Output destination
        io_layout.addWidget(QLabel("Output Dataset:"), 2, 0)
        self.output_name = QLineEdit()
        self.output_name.setText("Processed_Dataset_001")
        self.output_name.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 5px;
            padding: 5px;
        """)
        io_layout.addWidget(self.output_name, 2, 1)
        
        main_layout.addWidget(io_group)
        
        # Preprocessing steps
        preproc_group = QGroupBox("Preprocessing Steps")
        preproc_group.setStyleSheet("""
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
        
        preproc_layout = QVBoxLayout(preproc_group)
        
        # Normalization
        norm_layout = QHBoxLayout()
        self.norm_check = QCheckBox("Normalization")
        self.norm_check.setChecked(True)
        self.norm_check.setStyleSheet("""
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
        norm_layout.addWidget(self.norm_check)
        
        self.norm_method = QComboBox()
        self.norm_method.addItems(["Min-Max Scaling", "Standard (Z-score)", "Robust Scaling"])
        self.norm_method.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 5px;
            padding: 5px;
        """)
        norm_layout.addWidget(self.norm_method)
        preproc_layout.addLayout(norm_layout)
        
        # Missing value imputation
        impute_layout = QHBoxLayout()
        self.impute_check = QCheckBox("Missing Value Imputation")
        self.impute_check.setChecked(True)
        self.impute_check.setStyleSheet("""
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
        impute_layout.addWidget(self.impute_check)
        
        self.impute_method = QComboBox()
        self.impute_method.addItems(["Mean", "Median", "Most Frequent", "Constant"])
        self.impute_method.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 5px;
            padding: 5px;
        """)
        impute_layout.addWidget(self.impute_method)
        preproc_layout.addLayout(impute_layout)
        
        # Feature selection
        feature_layout = QHBoxLayout()
        self.feature_check = QCheckBox("Feature Selection")
        self.feature_check.setStyleSheet("""
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
        feature_layout.addWidget(self.feature_check)
        
        self.feature_method = QComboBox()
        self.feature_method.addItems(["PCA", "Select K Best", "Recursive Feature Elimination"])
        self.feature_method.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 5px;
            padding: 5px;
        """)
        feature_layout.addWidget(self.feature_method)
        preproc_layout.addLayout(feature_layout)
        
        # Data augmentation
        augment_layout = QHBoxLayout()
        self.augment_check = QCheckBox("Data Augmentation")
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
        augment_layout.addWidget(self.augment_check)
        
        self.augment_factor = QSpinBox()
        self.augment_factor.setRange(1, 10)
        self.augment_factor.setValue(2)
        self.augment_factor.setPrefix("Factor: ")
        self.augment_factor.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 5px;
            padding: 5px;
        """)
        augment_layout.addWidget(self.augment_factor)
        preproc_layout.addLayout(augment_layout)
        
        # Class balancing
        balance_layout = QHBoxLayout()
        self.balance_check = QCheckBox("Class Balancing")
        self.balance_check.setStyleSheet("""
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
        balance_layout.addWidget(self.balance_check)
        
        self.balance_method = QComboBox()
        self.balance_method.addItems(["SMOTE", "Random Oversampling", "Random Undersampling"])
        self.balance_method.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 5px;
            padding: 5px;
        """)
        balance_layout.addWidget(self.balance_method)
        preproc_layout.addLayout(balance_layout)
        
        main_layout.addWidget(preproc_group)
        
        # Train/validation/test split
        split_group = QGroupBox("Dataset Split")
        split_group.setStyleSheet("""
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
        
        split_layout = QGridLayout(split_group)
        
        # Train percentage
        split_layout.addWidget(QLabel("Training:"), 0, 0)
        self.train_pct = QSpinBox()
        self.train_pct.setRange(50, 90)
        self.train_pct.setValue(70)
        self.train_pct.setSuffix("%")
        self.train_pct.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 5px;
            padding: 5px;
        """)
        split_layout.addWidget(self.train_pct, 0, 1)
        
        # Validation percentage
        split_layout.addWidget(QLabel("Validation:"), 0, 2)
        self.val_pct = QSpinBox()
        self.val_pct.setRange(5, 30)
        self.val_pct.setValue(15)
        self.val_pct.setSuffix("%")
        self.val_pct.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 5px;
            padding: 5px;
        """)
        split_layout.addWidget(self.val_pct, 0, 3)
        
        # Test percentage
        split_layout.addWidget(QLabel("Test:"), 0, 4)
        self.test_pct = QSpinBox()
        self.test_pct.setRange(5, 30)
        self.test_pct.setValue(15)
        self.test_pct.setSuffix("%")
        self.test_pct.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 5px;
            padding: 5px;
        """)
        split_layout.addWidget(self.test_pct, 0, 5)
        
        # Stratify option
        self.stratify_check = QCheckBox("Stratified Split (preserve class distribution)")
        self.stratify_check.setChecked(True)
        self.stratify_check.setStyleSheet("""
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
        split_layout.addWidget(self.stratify_check, 1, 0, 1, 6)
        
        main_layout.addWidget(split_group)
        
        # Execution controls
        controls_layout = QHBoxLayout()
        
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
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #3498DB;
                border-radius: 4px;
            }
        """)
        controls_layout.addWidget(self.progress_bar)
        
        # Process button
        self.process_button = QPushButton("Process Dataset")
        self.process_button.setStyleSheet("""
            QPushButton {
                background-color: #27AE60;
                color: white;
                border-radius: 5px;
                padding: 10px 15px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2ECC71;
            }
            QPushButton:pressed {
                background-color: #1C8348;
            }
        """)
        controls_layout.addWidget(self.process_button)
        
        main_layout.addLayout(controls_layout)
        main_layout.addStretch()

class DatasetPanel(QWidget):
    """Panel for dataset management and exploration"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        """Initialize the dataset panel UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)
        
        # Header area
        header_layout = QHBoxLayout()
        
        # Title
        title = QLabel("Dataset Management")
        title.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #3498DB;
            margin-bottom: 10px;
        """)
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Dataset selector
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(QLabel("Active Dataset:"))
        
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems([
            "MNIST_digits", 
            "Memory_ConversationData_2023", 
            "Synthetic_TestData", 
            "Custom_MemoryFeatures"
        ])
        self.dataset_combo.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 5px;
            padding: 8px;
            min-width: 200px;
        """)
        dataset_layout.addWidget(self.dataset_combo)
        
        header_layout.addLayout(dataset_layout)
        
        # Import button
        self.import_button = QPushButton("Import New")
        self.import_button.setStyleSheet("""
            QPushButton {
                background-color: #2980B9;
                color: white;
                border-radius: 5px;
                padding: 8px 15px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3498DB;
            }
        """)
        header_layout.addWidget(self.import_button)
        
        main_layout.addLayout(header_layout)
        
        # Tabs for different dataset functions
        tabs = QTabWidget()
        tabs.setStyleSheet("""
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
        
        # Overview tab
        overview_tab = QWidget()
        overview_layout = QVBoxLayout(overview_tab)
        
        # Dataset stats
        stats_widget = DatasetStatsWidget()
        overview_layout.addWidget(stats_widget)
        
        tabs.addTab(overview_tab, "Overview")
        
        # Explore tab
        explore_tab = QWidget()
        explore_layout = QVBoxLayout(explore_tab)
        
        # Explorer widget
        explorer_widget = DataExplorerWidget()
        explore_layout.addWidget(explorer_widget)
        
        tabs.addTab(explore_tab, "Explore Data")
        
        # Prepare tab
        prepare_tab = QWidget()
        prepare_layout = QVBoxLayout(prepare_tab)
        
        # Data preparation widget
        prep_widget = DataPrepWidget()
        prepare_layout.addWidget(prep_widget)
        
        tabs.addTab(prepare_tab, "Prepare & Process")
        
        main_layout.addWidget(tabs)
        
        # Footer status bar
        footer_layout = QHBoxLayout()
        
        self.status_label = QLabel("Ready. Active dataset contains 1024 samples.")
        self.status_label.setStyleSheet("color: #95A5A6; font-size: 12px;")
        footer_layout.addWidget(self.status_label)
        
        footer_layout.addStretch()
        
        # Refresh button
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setStyleSheet("""
            QPushButton {
                background-color: #2C3E50;
                color: white;
                border-radius: 5px;
                padding: 5px 10px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #34495E;
            }
        """)
        footer_layout.addWidget(self.refresh_button)
        
        main_layout.addLayout(footer_layout) 