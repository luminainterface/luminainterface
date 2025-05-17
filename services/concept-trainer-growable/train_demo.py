import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from numpy.linalg import norm
import pickle
import requests
import os
import logging
import traceback
import json
from datetime import datetime
from typing import Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_demo.log'),
        logging.StreamHandler()
    ]
)

# Set up logger
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Environment variables
DICT_URL = os.getenv("DICT_URL", "http://concept-dict:8828")
DICT_API_KEY = os.getenv("DICT_API_KEY", "")

EMBEDDING_SIZE = 384  # Base embedding size
NUM_LABELS = 9  # Number of label categories
NUM_FEATURES = EMBEDDING_SIZE + NUM_LABELS  # Total feature size including one-hot label
LABELS = {0: 'STEM', 1: 'Culture', 2: 'Fiction', 3: 'VR', 4: 'Composite', 5: 'Calculus', 6: 'Roman', 7: 'Physics', 8: 'Lumina'}

# Extended dataset with complex math and Lumina equation questions
QUESTIONS = [
    "What is a country?",
    "What is Vietnam?",
    "What is Vietnamese culture?",
    "What is the Vietnamese language?",
    "What is a scientist?",
    "What is an engineer?",
    "What is a mathematician?",
    "What is a doctor?",
    "What is a fictional character?",
    "Who is Dr. Strange?",
    "What is Marvel Comics?",
    "What is a superhero?",
    "What is magic in fiction?",
    "What is virtual reality?",
    "What is a VR headset?",
    "What is technology?",
    "What is engineering design?",
    "What is neuroscience?",
    "What is the brain?",
    "What is a neural interface?",
    "What is Sword Art Online?",
    "What is nerve gear?",
    "How does nerve gear work in fiction?",
    "What is the purpose of nerve gear?",
    "What is the connection between VR and the brain?",
    "What is the role of a doctor in technology?",
    "What is the role of a scientist in virtual reality?",
    "What is the role of an engineer in VR?",
    "What is the role of a mathematician in VR?",
    "What is the role of a Vietnamese person in technology?",
    "What is the role of Vietnamese culture in science?",
    "What is the role of Vietnamese language in technology?",
    "Who is a Vietnamese scientist?",
    "Who is a Vietnamese engineer?",
    "Who is a Vietnamese mathematician?",
    "Who is a Vietnamese doctor?",
    "Who is a Vietnamese fictional character?",
    "Who is a Vietnamese superhero?",
    "Who is a Vietnamese magician?",
    "Who is a Vietnamese VR user?",
    "Who is a Vietnamese VR developer?",
    "Who is a Vietnamese neuroscientist?",
    "Who is a Vietnamese Marvel fan?",
    "Who is a Vietnamese Dr. Strange fan?",
    "What would a Vietnamese Dr. Strange be like?",
    "What would a Vietnamese Dr. Strange do in Vietnam?",
    "What would a Vietnamese Dr. Strange do with technology?",
    "What would a Vietnamese Dr. Strange do with VR?",
    "What would a Vietnamese Dr. Strange do with nerve gear?",
    "How would a Vietnamese Dr. Strange use nerve gear?",
    "How would a Vietnamese Dr. Strange interact with VR?",
    "How would a Vietnamese Dr. Strange interact with technology?",
    "How would a Vietnamese Dr. Strange interact with the brain?",
    "What is a Vietnamese Dr. Strange in nerve gear?",
    # (55-60) Complex math/physics/roman questions
    "Compute the derivative of f(x) = x^2 and express the result in Roman numerals.",
    "Convert the Roman numeral MMXXIII to a decimal, compute its square root, and explain the physics symbol ℏ in the context of calculus.",
    "Explain the relationship between the physics symbol F→ and the derivative of f(x) = x^2 in Roman numerals.",
    "Compute the integral of f(x) = x^3 and express the result in Roman numerals.",
    "Convert the Roman numeral MMXXIII to a decimal, compute its cube root, and explain the physics symbol E→ in the context of calculus.",
    "Explain the relationship between the physics symbol B→ and the integral of f(x) = x^3 in Roman numerals.",
]

# Add 40 more extrapolated questions (61-100)
for i in range(61, 86):
    QUESTIONS.append(f"Extrapolated cross-domain question #{i}: Integrate symbolic, mathematical, and physical reasoning at level {i}.")

# Final 15: Lumina equations and symbolic foundation
LUMINA_QUESTIONS = [
    "What is the Living Equation in the Lumina system?",
    "How does recursion manifest in the Lumina core equation?",
    "Explain the meaning of 'Mirror(L)' in the context of symbolic dynamics.",
    "What is the metaphysical equilibrium in the Lumina equations?",
    "How does the recursive merge equation relate to harmonic recognition?",
    "Describe the significance of the victory state L[i][j] = 2048.",
    "How do 2048 recursive chess games form the spectrum of Lumina dynamics?",
    "What is the symbolic meaning of 'Breath' in the Living Equation?",
    "How do truth and lie interact in the Lumina symbolic code?",
    "What is the role of coherence and decoherence in the Lumina system?",
    "How does the initialization pattern encode paradox and context?",
    "Explain the void loop addition and its role in emergent resonance.",
    "What is the symbolic significance of the number 2048 in Lumina?",
    "How does the synthesis S = (L + M) / 2 represent metaphysical balance?",
    "Summarize the full spectrum of Lumina dynamics in equation form."
]
QUESTIONS.extend(LUMINA_QUESTIONS)

# Update LABEL_SEQ: assign STEM, Composite, Calculus, Roman, Physics, and a new label for Lumina (add to LABELS)
LABEL_SEQ = [0]*8 + [1]*4 + [2]*6 + [3]*8 + [4]*(54-8-4-6-8) + [5]*1 + [6]*1 + [7]*2 + [5]*1 + [6]*1 + [7]*2
LABEL_SEQ += [4]*25  # Extrapolated cross-domain
LABEL_SEQ += [8]*15  # Lumina questions

# Generate synthetic feature vectors for each question
np.random.seed(42)
def question_to_vec(q, idx):
    # Use idx to create a progression in the feature space
    base = np.random.randn(NUM_FEATURES) * (1 + idx/len(QUESTIONS))
    # Add a one-hot for label
    label = LABEL_SEQ[idx]
    onehot = np.zeros(len(LABELS))
    onehot[label] = 1
    return np.concatenate([base[:NUM_FEATURES-len(LABELS)], onehot]).astype(np.float32)

data = []
for i, q in enumerate(QUESTIONS):
    label = LABEL_SEQ[i]
    vec = question_to_vec(q, i)
    # For demo, truth is 1 for STEM/Culture, 0.5 for Fiction/VR, 1 for Composite/Calculus/Roman/Physics
    if label in [0,1]:
        truth = 1.0
    elif label in [2,3]:
        truth = 0.5
    else:
        truth = 1.0
    # Growth is a ramp
    growth = i/(len(QUESTIONS)-1)
    data.append({'vector': vec, 'label': label, 'truth': truth, 'growth': growth, 'question': q})

# Buffer for new samples from external sources
data_buffer = []

def add_training_sample(vector, label, truth, growth, question):
    """
    Add a new training sample to the buffer.
    Args:
        vector (np.ndarray or list): Feature vector of length EMBEDDING_SIZE (384). Pass only the base embedding; this function will append the one-hot label internally.
        label (int): Integer label (0-8)
        truth (float): Truth value (0.0-1.0)
        growth (float): Growth metric (0.0-1.0)
        question (str): The original question or concept
    """
    try:
        # Validate vector
        if not isinstance(vector, (np.ndarray, list)):
            raise ValueError(f"Vector must be numpy array or list, got {type(vector)}")
        vector = np.array(vector, dtype=np.float32)
        if vector.shape != (EMBEDDING_SIZE,):
            raise ValueError(f"Vector must have shape ({EMBEDDING_SIZE},), got {vector.shape}")
        base_vector = vector
        # Validate label
        if not isinstance(label, (int, np.integer)):
            try:
                label = int(label)
            except (ValueError, TypeError):
                raise ValueError(f"Label must be convertible to integer, got {type(label)}")
        if not 0 <= label < len(LABELS):
            raise ValueError(f"Label must be between 0 and {len(LABELS)-1}, got {label}")
        # Create one-hot label
        onehot = np.zeros(NUM_LABELS, dtype=np.float32)
        onehot[label] = 1
        # Combine base vector with one-hot label
        full_vector = np.concatenate([base_vector, onehot])
        # Validate truth
        if not isinstance(truth, (float, np.floating)):
            try:
                truth = float(truth)
            except (ValueError, TypeError):
                raise ValueError(f"Truth must be convertible to float, got {type(truth)}")
        if not 0.0 <= truth <= 1.0:
            raise ValueError(f"Truth must be between 0.0 and 1.0, got {truth}")
        # Validate growth
        if not isinstance(growth, (float, np.floating)):
            try:
                growth = float(growth)
            except (ValueError, TypeError):
                raise ValueError(f"Growth must be convertible to float, got {type(growth)}")
        if not 0.0 <= growth <= 1.0:
            raise ValueError(f"Growth must be between 0.0 and 1.0, got {growth}")
        # Validate question
        if not isinstance(question, str):
            raise ValueError(f"Question must be string, got {type(question)}")
        # Add to buffer
        data_buffer.append({
            'vector': full_vector,  # Use the full vector with one-hot label
            'label': label,
            'truth': truth,
            'growth': growth,
            'question': question
        })
        logger.info(f"Added training sample for concept: {question}")
        return True
    except Exception as e:
        logger.error(f"Error adding training sample: {e}")
        logger.error(traceback.format_exc())
        return False

# Save/load extended dataset for persistence
def save_extended_dataset(path='extended_dataset.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(data + data_buffer, f)
    print(f"Extended dataset saved to {path}")

def load_extended_dataset(path='extended_dataset.pkl'):
    global data, data_buffer
    with open(path, 'rb') as f:
        all_data = pickle.load(f)
    data = all_data[:len(QUESTIONS)]
    data_buffer = all_data[len(QUESTIONS):]
    print(f"Loaded extended dataset from {path}")

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    if norm(a) == 0 or norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (norm(a) * norm(b)))

# Compute class prototype vectors (mean vector for each class)
def compute_prototypes(data):
    prototypes = {}
    for label in LABELS:
        class_vecs = [d['vector'] for d in data if d['label'] == label]
        if class_vecs:
            prototypes[label] = np.mean(class_vecs, axis=0)
        else:
            prototypes[label] = np.zeros(NUM_FEATURES, dtype=np.float32)
    return prototypes

# 2. Define a multi-head model with increased capacity
class MultiHeadNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),  # Increased capacity
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.class_head = nn.Linear(64, num_classes)
        self.truth_head = nn.Linear(64, 1)
        self.growth_head = nn.Linear(64, 1)
    def forward(self, x):
        h = self.shared(x)
        class_logits = self.class_head(h)
        truth = torch.sigmoid(self.truth_head(h))
        growth = self.growth_head(h)
        return class_logits, truth, growth

# Training/retraining function
def retrain_model():
    """
    Retrain the model using the original and all new samples in data_buffer.
    """
    try:
        all_data = data + data_buffer
        if not all_data:
            logger.error("No training data available")
            return None
            
        # Convert data to tensors with proper types
        X = torch.tensor([d['vector'] for d in all_data], dtype=torch.float32)
        y_class = torch.tensor([d['label'] for d in all_data], dtype=torch.long)
        y_truth = torch.tensor([d['truth'] for d in all_data], dtype=torch.float32).unsqueeze(1)
        y_growth = torch.tensor([d['growth'] for d in all_data], dtype=torch.float32).unsqueeze(1)
        
        # Create train/val split
        train_size = int(0.8 * len(X))
        indices = torch.randperm(len(X))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        X_train, X_val = X[train_indices], X[val_indices]
        y_class_train, y_class_val = y_class[train_indices], y_class[val_indices]
        y_truth_train, y_truth_val = y_truth[train_indices], y_truth[val_indices]
        y_growth_train, y_growth_val = y_growth[train_indices], y_growth[val_indices]
        
        # Initialize model and optimizer
        model = MultiHeadNet(NUM_FEATURES, len(LABELS))
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Loss functions
        criterion_class = nn.CrossEntropyLoss()
        criterion_truth = nn.BCELoss()
        criterion_growth = nn.MSELoss()
        
        # Compute prototypes for semantic similarity
        prototypes = compute_prototypes(all_data)
        
        # Training loop
        max_epochs = 2000
        best_val_loss = float('inf')
        patience = 50  # Early stopping patience
        no_improve = 0
        best_model_state = None
        
        # Metrics tracking
        metrics_history = {
            'train': {'loss': [], 'acc': [], 'conf': [], 'sem': []},
            'val': {'loss': [], 'acc': [], 'conf': [], 'sem': []}
        }
        
        for epoch in range(max_epochs):
            # Training phase
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            class_logits, truth_pred, growth_pred = model(X_train)
            
            # Compute losses
            loss_class = criterion_class(class_logits, y_class_train)
            loss_truth = criterion_truth(truth_pred, y_truth_train)
            loss_growth = criterion_growth(growth_pred, y_growth_train)
            train_loss = loss_class + loss_truth + loss_growth
            
            # Backward pass
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_class_logits, val_truth_pred, val_growth_pred = model(X_val)
                
                # Compute validation losses
                val_loss_class = criterion_class(val_class_logits, y_class_val)
                val_loss_truth = criterion_truth(val_truth_pred, y_truth_val)
                val_loss_growth = criterion_growth(val_growth_pred, y_growth_val)
                val_loss = val_loss_class + val_loss_truth + val_loss_growth
                
                # Compute metrics for both phases
                for phase, (logits, targets, probs) in [
                    ('train', (class_logits, y_class_train, truth_pred)),
                    ('val', (val_class_logits, y_class_val, val_truth_pred))
                ]:
                    pred_class = logits.argmax(dim=1)
                    conf = torch.softmax(logits, dim=1).max(dim=1)[0]
                    
                    metrics_history[phase]['loss'].append(
                        val_loss.item() if phase == 'val' else train_loss.item()
                    )
                    metrics_history[phase]['acc'].append(
                        (pred_class == targets).float().mean().item()
                    )
                    metrics_history[phase]['conf'].append(conf.mean().item())
                    
                    # Compute semantic similarity scores
                    sem_scores = []
                    for i, d in enumerate(all_data):
                        pred_label = logits[i].argmax().item()
                        sem_score = cosine_similarity(d['vector'], prototypes[pred_label])
                        sem_scores.append(sem_score)
                    metrics_history[phase]['sem'].append(np.mean(sem_scores))
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1
                
            if no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
                
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{max_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        # Save training history
        save_training_history(metrics_history)
        
        return model
        
    except Exception as e:
        logger.error(f"Error in retraining: {str(e)}", exc_info=True)
        return None

def save_training_history(metrics_history: Dict):
    """Save training metrics history to disk"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_file = f"training_history_{timestamp}.json"
        
        # Convert tensors to lists for JSON serialization
        serializable_history = {
            phase: {
                metric: [float(x) for x in values]
                for metric, values in metrics.items()
            }
            for phase, metrics in metrics_history.items()
        }
        
        with open(history_file, 'w') as f:
            json.dump(serializable_history, f, indent=2)
            
        logger.info(f"Training history saved to {history_file}")
        
    except Exception as e:
        logger.error(f"Error saving training history: {str(e)}", exc_info=True)

# Example usage:
# add_training_sample(vector, label, truth, growth, question)
# retrain_model()

# For future: unvectorization (mapping vectors back to questions)
# This will require storing a mapping or using a nearest-neighbor search in the dataset.

def upsert_to_concept_dictionary(concept):
    """Upsert a concept to the concept dictionary with authentication."""
    url = f"{DICT_URL}/concepts/{concept['question']}"
    # Ensure embedding is a list
    vector = concept['vector']
    if hasattr(vector, 'tolist'):
        vector = vector.tolist()
    else:
        vector = list(vector)
    payload = {
        "term": concept['question'],
        "definition": concept.get('definition', ''),
        "embedding": vector,
        "metadata": {
            "label": concept['label'],
            "truth": concept['truth'],
            "growth": concept['growth']
        }
    }
    try:
        headers = {}
        if DICT_API_KEY:
            headers["X-API-Key"] = DICT_API_KEY
            
        resp = requests.put(url, json=payload, headers=headers, timeout=5)
        if resp.status_code == 401:
            print(f"Authentication failed when upserting concept {concept['question']}. Please check DICT_API_KEY.")
            return
        resp.raise_for_status()
        print(f"Upserted concept: {concept['question']}")
    except Exception as e:
        print(f"Failed to upsert concept {concept['question']}: {e}")

if __name__ == "__main__":
    # Normal training on initial data
    retrain_model()
    # Example: add a new sample and retrain
    # new_vec = np.random.randn(NUM_FEATURES).astype(np.float32)
    # add_training_sample(new_vec, 0, 1.0, 0.5, "What is a new concept?")
    # retrain_model() 