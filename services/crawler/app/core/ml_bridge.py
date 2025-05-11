from .ml_priority import train_priority_model
from .ml_output import train_output_model

class MLBridge:
    """
    Bridge for cross-training between crawl prioritization and output optimization models.
    Collects training data, coordinates training, and passes cross-features between models.
    """
    def __init__(self):
        self.priority_training_data = []  # List of dicts for priority model
        self.output_training_data = []    # List of dicts for output model
        self.priority_scores = []         # Last output from priority model
        self.output_scores = []           # Last output from output model

    def add_priority_sample(self, sample, output_score=None):
        """Add a training sample for the priority model, optionally with output model score."""
        self.priority_training_data.append(sample)
        if output_score is not None:
            self.output_scores.append(output_score)

    def add_output_sample(self, sample, priority_score=None):
        """Add a training sample for the output model, optionally with priority model score."""
        self.output_training_data.append(sample)
        if priority_score is not None:
            self.priority_scores.append(priority_score)

    def cross_train(self, epochs=10, lr=1e-3):
        """
        Perform cross-training: train each model using the other's scores as features.
        """
        print("[MLBridge] Starting cross-training...")
        # Train priority model with output scores as extra feature
        priority_model = train_priority_model(self.priority_training_data, output_scores=self.output_scores, epochs=epochs, lr=lr)
        # Train output model with priority scores as extra feature
        output_model = train_output_model(self.output_training_data, crawl_priority_scores=self.priority_scores, epochs=epochs, lr=lr)
        print("[MLBridge] Cross-training complete.")
        return priority_model, output_model

    def clear(self):
        """Clear all stored training data and scores."""
        self.priority_training_data.clear()
        self.output_training_data.clear()
        self.priority_scores.clear()
        self.output_scores.clear() 