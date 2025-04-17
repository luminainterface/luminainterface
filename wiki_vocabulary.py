import os
import json
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

logger = logging.getLogger(__name__)

class WikiVocabulary:
    def __init__(self, cache_dir="data/wiki_vocabulary", max_features=1024):
        self.cache_dir = cache_dir
        self.vocabulary = {}
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        self.fitted = False
        os.makedirs(cache_dir, exist_ok=True)
        self.load_vocabulary()
        self.load_vectorizer()

    def load_vocabulary(self):
        vocab_file = os.path.join(self.cache_dir, "vocabulary.json")
        if os.path.exists(vocab_file):
            try:
                with open(vocab_file, 'r', encoding='utf-8') as f:
                    self.vocabulary = json.load(f)
                logger.info(f"Loaded {len(self.vocabulary)} articles from vocabulary")
                
                # Fit vectorizer on loaded content if we have articles
                if self.vocabulary:
                    self.fit_vectorizer()
            except Exception as e:
                logger.error(f"Error loading vocabulary: {e}")

    def save_vocabulary(self):
        vocab_file = os.path.join(self.cache_dir, "vocabulary.json")
        try:
            with open(vocab_file, 'w', encoding='utf-8') as f:
                json.dump(self.vocabulary, f)
            logger.info(f"Saved {len(self.vocabulary)} articles to vocabulary")
            self.save_vectorizer()
        except Exception as e:
            logger.error(f"Error saving vocabulary: {e}")

    def load_vectorizer(self):
        """Load the fitted vectorizer if it exists"""
        vectorizer_file = os.path.join(self.cache_dir, "vectorizer.pkl")
        if os.path.exists(vectorizer_file):
            try:
                with open(vectorizer_file, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                self.fitted = True
                logger.info("Loaded fitted vectorizer")
            except Exception as e:
                logger.error(f"Error loading vectorizer: {e}")
                self.fitted = False

    def save_vectorizer(self):
        """Save the fitted vectorizer"""
        if self.fitted:
            vectorizer_file = os.path.join(self.cache_dir, "vectorizer.pkl")
            try:
                with open(vectorizer_file, 'wb') as f:
                    pickle.dump(self.vectorizer, f)
                logger.info("Saved fitted vectorizer")
            except Exception as e:
                logger.error(f"Error saving vectorizer: {e}")

    def fit_vectorizer(self):
        """Fit the vectorizer on all available content"""
        try:
            content_list = [article['content'] for article in self.vocabulary.values()]
            if content_list:
                self.vectorizer.fit(content_list)
                self.fitted = True
                logger.info("Fitted vectorizer on vocabulary content")
        except Exception as e:
            logger.error(f"Error fitting vectorizer: {e}")
            self.fitted = False

    def add_article(self, title, content, metadata=None):
        """Add an article and update the vectorizer"""
        if title not in self.vocabulary:
            self.vocabulary[title] = {
                'content': content,
                'metadata': metadata or {}
            }
            
            # Update vectorizer with new content
            try:
                if not self.fitted:
                    self.vectorizer.fit([content])
                    self.fitted = True
                else:
                    # Could optionally refit on all content, but skip for efficiency
                    pass
            except Exception as e:
                logger.error(f"Error updating vectorizer: {e}")
            
            self.save_vocabulary()
            return True
        return False

    def vectorize_text(self, text):
        """Convert text to vector using TF-IDF"""
        try:
            if not self.fitted:
                self.vectorizer.fit([text])
                self.fitted = True
                
            vector = self.vectorizer.transform([text]).toarray()[0]
            
            # Ensure vector has exactly max_features dimensions
            if len(vector) < self.max_features:
                vector = np.pad(vector, (0, self.max_features - len(vector)))
            elif len(vector) > self.max_features:
                vector = vector[:self.max_features]
                
            return vector
            
        except Exception as e:
            logger.error(f"Error vectorizing text: {e}")
            # Return zero vector as fallback
            return np.zeros(self.max_features)

    def get_vocabulary_size(self):
        return len(self.vocabulary)

    def get_vector_size(self):
        """Return the size of vectors produced by the vectorizer"""
        return self.max_features 