import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CrawlPriorityModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Output between 0 and 1
        return x

def score_crawl_candidates(candidates, target_vector=None, model=None):
    """
    candidates: list of dicts, each with keys: 'vector', 'usage', 'last_crawled'
    target_vector: np.array or list, the goal/query vector
    model: a CrawlPriorityModel instance
    Returns: list of (candidate, score)
    """
    if model is None:
        model = CrawlPriorityModel()
        model.eval()
    if target_vector is None:
        target_vector = np.zeros(len(candidates[0]['vector']))

    now = int(torch.tensor([]).new_zeros(1).item())
    features = []
    for c in candidates:
        # Feature 1: cosine similarity to target
        v = np.array(c['vector'])
        t = np.array(target_vector)
        sim = float(np.dot(v, t) / (np.linalg.norm(v) * np.linalg.norm(t) + 1e-8))
        # Feature 2: usage (normalize to [0,1])
        usage = float(c.get('usage', 0)) / 100.0
        # Feature 3: recency (how long since last crawled, normalized)
        last_crawled = float(c.get('last_crawled', 0))
        recency = 1.0 / (1.0 + (now - last_crawled) / 3600.0) if last_crawled > 0 else 1.0
        features.append([sim, usage, recency])
    x = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        scores = model(x).squeeze().numpy()
    return list(zip(candidates, scores))

def train_priority_model(training_data, output_scores=None, epochs=10, lr=1e-3):
    """
    Placeholder for future training of the CrawlPriorityModel.
    Args:
        training_data: list of dicts with keys 'vector', 'usage', 'last_crawled', 'label' (desired priority score)
        output_scores: optional list of floats, output model scores for each candidate (cross-cache feature)
        epochs: number of training epochs
        lr: learning rate
    Returns:
        Trained CrawlPriorityModel
    Notes:
        If output_scores is provided, it can be appended as an extra feature to each training sample.
    """
    model = CrawlPriorityModel(input_dim=4 if output_scores is not None else 3)
    # TODO: Implement actual training loop
    # Example: for epoch in range(epochs): ...
    # If output_scores is not None, concatenate to feature vectors
    print("[train_priority_model] Training not yet implemented. Cross-cache output_scores received:" , output_scores is not None)
    return model 