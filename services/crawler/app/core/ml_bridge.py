from .ml_priority import train_priority_model
from .ml_output import train_output_model
import torch
import torch.optim as optim
import torch.nn as nn
import json
from datetime import datetime

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
        self.metrics_history = {
            'priority': {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []},
            'output': {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        }

    def add_priority_sample(self, sample, output_score=None):
        """Add a training sample for the priority model, optionally with output model score."""
        if output_score is not None:
            sample['output_score'] = output_score
        self.priority_training_data.append(sample)

    def add_output_sample(self, sample, priority_score=None):
        """Add a training sample for the output model, optionally with priority model score."""
        if priority_score is not None:
            sample['priority_score'] = priority_score
        self.output_training_data.append(sample)

    def _prepare_batch(self, data, batch_size=32):
        """Prepare batches for training"""
        indices = torch.randperm(len(data))
        for i in range(0, len(data), batch_size):
            batch_indices = indices[i:i + batch_size]
            yield [data[j] for j in batch_indices]

    def _train_model(self, model, train_data, val_data, epochs, lr, model_type):
        """Train a single model with validation"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        criterion = nn.BCEWithLogitsLoss() if model_type == 'priority' else nn.MSELoss()
        best_val_loss = float('inf')
        patience = 10
        no_improve = 0
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in self._prepare_batch(train_data):
                optimizer.zero_grad()
                
                # Prepare batch data
                vectors = torch.tensor([d['vector'] for d in batch], dtype=torch.float32, device=device)
                labels = torch.tensor([d['label'] for d in batch], dtype=torch.float32, device=device)
                
                # Add cross-features if available
                if model_type == 'priority' and 'output_score' in batch[0]:
                    output_scores = torch.tensor([d['output_score'] for d in batch], dtype=torch.float32, device=device)
                    vectors = torch.cat([vectors, output_scores.unsqueeze(1)], dim=1)
                elif model_type == 'output' and 'priority_score' in batch[0]:
                    priority_scores = torch.tensor([d['priority_score'] for d in batch], dtype=torch.float32, device=device)
                    vectors = torch.cat([vectors, priority_scores.unsqueeze(1)], dim=1)
                
                # Forward pass
                outputs = model(vectors)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                if model_type == 'priority':
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    train_correct += (preds == labels).sum().item()
                    train_total += labels.size(0)
            
            avg_train_loss = train_loss / len(train_data)
            train_acc = train_correct / train_total if model_type == 'priority' else None
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in self._prepare_batch(val_data):
                    vectors = torch.tensor([d['vector'] for d in batch], dtype=torch.float32, device=device)
                    labels = torch.tensor([d['label'] for d in batch], dtype=torch.float32, device=device)
                    
                    # Add cross-features if available
                    if model_type == 'priority' and 'output_score' in batch[0]:
                        output_scores = torch.tensor([d['output_score'] for d in batch], dtype=torch.float32, device=device)
                        vectors = torch.cat([vectors, output_scores.unsqueeze(1)], dim=1)
                    elif model_type == 'output' and 'priority_score' in batch[0]:
                        priority_scores = torch.tensor([d['priority_score'] for d in batch], dtype=torch.float32, device=device)
                        vectors = torch.cat([vectors, priority_scores.unsqueeze(1)], dim=1)
                    
                    outputs = model(vectors)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    if model_type == 'priority':
                        preds = (torch.sigmoid(outputs) > 0.5).float()
                        val_correct += (preds == labels).sum().item()
                        val_total += labels.size(0)
            
            avg_val_loss = val_loss / len(val_data)
            val_acc = val_correct / val_total if model_type == 'priority' else None
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Update metrics history
            self.metrics_history[model_type]['train_loss'].append(avg_train_loss)
            self.metrics_history[model_type]['val_loss'].append(avg_val_loss)
            if model_type == 'priority':
                self.metrics_history[model_type]['train_acc'].append(train_acc)
                self.metrics_history[model_type]['val_acc'].append(val_acc)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1
                
            if no_improve >= patience:
                logger.info(f"Early stopping triggered for {model_type} model after {epoch + 1} epochs")
                break
                
            # Log progress
            if (epoch + 1) % 5 == 0:
                log_msg = f"Epoch {epoch + 1}/{epochs} - {model_type} model - "
                log_msg += f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
                if model_type == 'priority':
                    log_msg += f", Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
                logger.info(log_msg)
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        return model

    def cross_train(self, epochs=10, lr=1e-3):
        """
        Perform cross-training: train each model using the other's scores as features.
        """
        logger.info("Starting cross-training...")
        
        # Create train/val splits
        def split_data(data):
            indices = torch.randperm(len(data))
            split_idx = int(0.8 * len(data))
            return data[:split_idx], data[split_idx:]
            
        priority_train, priority_val = split_data(self.priority_training_data)
        output_train, output_val = split_data(self.output_training_data)
        
        # Train priority model
        logger.info("Training priority model...")
        priority_model = train_priority_model(priority_train, output_scores=self.output_scores)
        priority_model = self._train_model(
            priority_model, priority_train, priority_val, epochs, lr, 'priority'
        )
        
        # Train output model
        logger.info("Training output model...")
        output_model = train_output_model(output_train, crawl_priority_scores=self.priority_scores)
        output_model = self._train_model(
            output_model, output_train, output_val, epochs, lr, 'output'
        )
        
        # Save training history
        self._save_training_history()
        
        logger.info("Cross-training complete.")
        return priority_model, output_model

    def _save_training_history(self):
        """Save training metrics history to disk"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            history_file = f"mlbridge_history_{timestamp}.json"
            
            with open(history_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
                
            logger.info(f"Training history saved to {history_file}")
            
        except Exception as e:
            logger.error(f"Error saving training history: {str(e)}", exc_info=True)

    def clear(self):
        """Clear all stored training data and scores."""
        self.priority_training_data.clear()
        self.output_training_data.clear()
        self.priority_scores.clear()
        self.output_scores.clear() 