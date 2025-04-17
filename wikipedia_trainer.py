import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import requests
from bs4 import BeautifulSoup
import re
import os
from datetime import datetime
import sqlite3

class WikipediaDataset(Dataset):
    def __init__(self, texts, labels, vectorizer):
        self.texts = texts
        self.labels = labels
        self.vectorizer = vectorizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert text to TF-IDF features
        features = self.vectorizer.transform([text]).toarray()[0]
        features = torch.FloatTensor(features)
        label = torch.LongTensor([label])[0]
        
        return features, label

class EnhancedWikipediaNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(EnhancedWikipediaNet, self).__init__()
        
        # Embedding layer to capture semantic relationships
        self.embedding = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(2048, num_heads=8, batch_first=True)
        
        # Feature extraction layers
        layers = []
        prev_size = 2048
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        self.feature_layers = nn.Sequential(*layers)
        
        # Classification head with skip connection
        self.classifier = nn.Sequential(
            nn.Linear(prev_size + 2048, 512),  # Skip connection from embedding
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Initial embedding
        embedded = self.embedding(x)
        
        # Self-attention (need to unsqueeze for batch dimension)
        attended, _ = self.attention(embedded.unsqueeze(1), 
                                   embedded.unsqueeze(1), 
                                   embedded.unsqueeze(1))
        attended = attended.squeeze(1)
        
        # Feature extraction
        features = self.feature_layers(attended)
        
        # Skip connection and classification
        combined = torch.cat([features, embedded], dim=1)
        output = self.classifier(combined)
        
        return output

def fetch_wikipedia_content(url):
    try:
        print(f"Fetching content from: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements first
        for element in soup.find_all(['table', 'style', 'script', 'sup', 'span.mw-editsection']):
            element.decompose()
        
        # Extract main content paragraphs
        content_div = soup.find('div', {'id': 'mw-content-text'})
        if not content_div:
            print(f"No content div found in {url}")
            return None
            
        # Find all paragraphs in the content
        paragraphs = content_div.find_all('p')
        if not paragraphs:
            print(f"No paragraphs found in {url}")
            return None
            
        # Combine paragraphs and clean text
        text = ' '.join(p.get_text() for p in paragraphs)
        text = re.sub(r'\[\d+\]', '', text)  # Remove reference numbers [1], [2], etc.
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = text.strip()
        
        if not text:
            print(f"Empty content from {url}")
            return None
            
        print(f"Successfully fetched content from {url} ({len(text)} characters)")
        return text
    except requests.exceptions.RequestException as e:
        print(f"Request error for {url}: {str(e)}")
        return None
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")
        return None

def process_wikipedia_links(file_path):
    try:
        print(f"\nProcessing file: {file_path}")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return [], []
            
        with open(file_path, 'r', encoding='utf-8') as f:
            links = [line.strip() for line in f if line.strip()]
            
        if not links:
            print(f"No links found in {file_path}")
            return [], []
            
        print(f"Found {len(links)} links in {file_path}")
        texts = []
        categories = []
        
        for i, link in enumerate(links, 1):
            print(f"\nProcessing link {i}/{len(links)}: {link}")
            text = fetch_wikipedia_content(link)
            if text:
                texts.append(text)
                # Extract category from filename
                category = os.path.basename(file_path).replace('.txt', '')
                categories.append(category)
                print(f"Successfully processed link {i}")
            else:
                print(f"Failed to process link {i}")
                
        print(f"\nSuccessfully processed {len(texts)} out of {len(links)} links from {file_path}")
        return texts, categories
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return [], []

def save_training_data(file_path, accuracy, loss, history):
    try:
        conn = sqlite3.connect('training_data.db')
        cursor = conn.cursor()
        
        # Save results
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''
            INSERT INTO training_results (file_path, accuracy, loss, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (file_path, accuracy, loss, timestamp))
        
        # Save history
        for epoch, epoch_loss, epoch_accuracy, lr in history:
            cursor.execute('''
                INSERT INTO training_history (file_path, epoch, loss, accuracy, learning_rate)
                VALUES (?, ?, ?, ?, ?)
            ''', (file_path, epoch, epoch_loss, epoch_accuracy, lr))
            
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving training data: {str(e)}")

def train_wikipedia_model(file_paths, hidden_sizes=[1024, 512, 256], batch_size=32, num_epochs=150):
    try:
        # Process all files
        all_texts = []
        all_categories = []
        
        for file_path in file_paths:
            texts, categories = process_wikipedia_links(file_path)
            all_texts.extend(texts)
            all_categories.extend(categories)
            
        if not all_texts:
            print("No valid Wikipedia content found")
            return
            
        # Convert categories to numerical labels
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(all_categories)
        
        # Enhanced text vectorization
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),  # Include phrases up to 3 words
            stop_words='english',
            max_df=0.95,  # Remove very common words
            min_df=2  # Remove very rare words
        )
        X = vectorizer.fit_transform(all_texts)
        
        # Create dataset and dataloader
        dataset = WikipediaDataset(all_texts, labels, vectorizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize enhanced model
        input_size = len(vectorizer.get_feature_names_out())
        num_classes = len(label_encoder.classes_)
        model = EnhancedWikipediaNet(input_size, hidden_sizes, num_classes)
        
        # Loss and optimizer with learning rate scheduling
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=5, 
                                                       verbose=True)
        
        # Training loop with improved monitoring
        history = []
        best_loss = float('inf')
        best_model_state = None
        patience = 15
        patience_counter = 0
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_texts, batch_labels in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_texts)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
                
            epoch_loss = total_loss / len(dataloader)
            epoch_accuracy = correct / total
            
            # Update learning rate
            scheduler.step(epoch_loss)
            
            # Early stopping with model saving
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
                    
            history.append((epoch + 1, epoch_loss, epoch_accuracy, optimizer.param_groups[0]['lr']))
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
            
        # Restore best model
        model.load_state_dict(best_model_state)
        
        # Save training data
        for file_path in file_paths:
            save_training_data(file_path, epoch_accuracy, epoch_loss, history)
            
        return model, vectorizer, label_encoder
        
    except Exception as e:
        print(f"Error in training: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    # All category files
    file_paths = [
        "science.txt",
        "history.txt",
        "technology.txt",
        "arts.txt",
        "philosophy.txt",
        "mathematics.txt"  # Added mathematics category
    ]
    
    # Use a larger model for more categories
    hidden_sizes = [1024, 768, 512]
    model, vectorizer, label_encoder = train_wikipedia_model(file_paths, hidden_sizes=hidden_sizes)
    
    if model:
        print("Training completed successfully")
        print("\nModel can predict the following categories:")
        for i, category in enumerate(label_encoder.classes_):
            print(f"{i+1}. {category}")
            
        # Save model and preprocessing objects
        torch.save(model.state_dict(), 'wikipedia_model.pth')
        # Save vectorizer and label encoder
        import pickle
        with open('wikipedia_preprocessor.pkl', 'wb') as f:
            pickle.dump((vectorizer, label_encoder), f)
    else:
        print("Training failed") 