import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class OutputOptimizationModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Output between 0 and 1
        return x

def score_outputs(candidates, context_vector=None, model=None):
    """
    candidates: list of dicts, each with keys: 'vector', 'context', 'prev_output', ...
    context_vector: np.array or list, the current context embedding
    model: an OutputOptimizationModel instance
    Returns: list of (candidate, score)
    """
    if model is None:
        model = OutputOptimizationModel()
        model.eval()
    if context_vector is None:
        context_vector = np.zeros(len(candidates[0]['vector']))

    features = []
    for c in candidates:
        v = np.array(c['vector'])
        ctx = np.array(context_vector)
        prev = np.array(c.get('prev_output', np.zeros_like(v)))
        # Example features: cosine similarity to context, previous output similarity, vector norm
        sim_to_ctx = float(np.dot(v, ctx) / (np.linalg.norm(v) * np.linalg.norm(ctx) + 1e-8))
        sim_to_prev = float(np.dot(v, prev) / (np.linalg.norm(v) * np.linalg.norm(prev) + 1e-8))
        norm = float(np.linalg.norm(v)) / 10.0  # normalize
        features.append([sim_to_ctx, sim_to_prev, norm] + [0.0]*(10-3))  # pad to input_dim
    x = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        scores = model(x).squeeze().numpy()
    return list(zip(candidates, scores))

def train_output_model(training_data, crawl_priority_scores=None, epochs=10, lr=1e-3):
    """
    Placeholder for future training of the OutputOptimizationModel.
    Args:
        training_data: list of dicts with keys 'vector', 'context', 'prev_output', 'label' (desired output score)
        crawl_priority_scores: optional list of floats, crawl priority model scores for each candidate (cross-cache feature)
        epochs: number of training epochs
        lr: learning rate
    Returns:
        Trained OutputOptimizationModel
    Notes:
        If crawl_priority_scores is provided, it can be appended as an extra feature to each training sample.
    """
    model = OutputOptimizationModel(input_dim=11 if crawl_priority_scores is not None else 10)
    # TODO: Implement actual training loop
    # If crawl_priority_scores is not None, concatenate to feature vectors
    print("[train_output_model] Training not yet implemented. Cross-cache crawl_priority_scores received:", crawl_priority_scores is not None)
    return model 