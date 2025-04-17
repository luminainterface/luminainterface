"""
Dream Archive Module for V7 Dream Mode

This module implements the Dream Archive component of the Dream Mode system,
which records and classifies dream content during dream states.
"""

import os
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
import uuid

# Set up logging
logger = logging.getLogger("lumina_v7.dream_archive")

class DreamArchive:
    """
    Records and classifies dream content
    
    Key features:
    - Persistent storage of dream records
    - Classification of dream types and content
    - Retrieval of past dreams by ID or criteria
    - Dream content search and filtering
    - Statistics about dream patterns over time
    """
    
    def __init__(self, data_dir: str = "data/v7/dreams", max_size: int = 100, 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Dream Archive
        
        Args:
            data_dir: Directory to store dream data
            max_size: Maximum number of dreams to store
            config: Configuration dictionary (optional)
        """
        # Default configuration
        self.config = {
            "auto_cleanup": True,          # Automatically remove oldest dreams when max reached
            "persist_frequency": 5,        # Number of new dreams before forced persistence
            "archive_file": "dreams.json", # Main archive file name
            "index_file": "dream_index.json",  # Index file name
            "dream_classification": True,   # Classify dreams by content
            "backup_archives": True,        # Create backups of archive files
            "max_backups": 3               # Maximum number of backup files to keep
        }
        
        # Update with custom config
        if config:
            self.config.update(config)
        
        # Dream storage settings
        self.data_dir = Path(data_dir)
        self.max_size = max(10, max_size)  # Minimum 10 dreams
        
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dreams collection
        self.dreams = []
        self.dream_index = {}
        self.new_dreams_count = 0
        
        # Statistics tracking
        self.stats = {
            "total_dreams": 0,
            "dreams_by_month": {},
            "most_common_topics": {},
            "insight_count": 0,
            "average_dream_duration": 0,
            "last_updated": None
        }
        
        # Locking
        self.archive_lock = threading.Lock()
        
        # Load existing dreams
        self._load_archive()
        
        logger.info(f"Dream Archive initialized with {len(self.dreams)} dreams")
    
    def _load_archive(self):
        """Load existing dreams from archive file"""
        archive_path = self.data_dir / self.config["archive_file"]
        index_path = self.data_dir / self.config["index_file"]
        
        try:
            # Load main archive if it exists
            if archive_path.exists():
                with open(archive_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Process archive data
                    if isinstance(data, dict):
                        # New format with metadata
                        self.dreams = data.get("dreams", [])
                        self.stats = data.get("stats", self.stats)
                    elif isinstance(data, list):
                        # Old format (just a list of dreams)
                        self.dreams = data
                        # Generate stats for old format
                        self._recalculate_stats()
                    
                    logger.info(f"Loaded {len(self.dreams)} dreams from archive")
            
            # Load index if it exists
            if index_path.exists():
                with open(index_path, 'r', encoding='utf-8') as f:
                    self.dream_index = json.load(f)
                    logger.info(f"Loaded dream index with {len(self.dream_index)} entries")
            else:
                # Build index from dreams
                self._rebuild_index()
                
        except Exception as e:
            logger.error(f"Error loading dream archive: {e}")
            # Initialize empty archive
            self.dreams = []
            self.dream_index = {}
            self._recalculate_stats()
    
    def _save_archive(self, force: bool = False):
        """
        Save dreams to archive file
        
        Args:
            force: Force save even if new_dreams_count is below threshold
        """
        # Skip if no new dreams and not forced
        if self.new_dreams_count == 0 and not force:
            return
        
        # Use lock to prevent concurrent saves
        with self.archive_lock:
            try:
                # Update statistics
                self._recalculate_stats()
                
                archive_path = self.data_dir / self.config["archive_file"]
                index_path = self.data_dir / self.config["index_file"]
                
                # Create backup if enabled
                if self.config["backup_archives"] and archive_path.exists():
                    self._create_backup(archive_path)
                
                # Save main archive
                archive_data = {
                    "dreams": self.dreams,
                    "stats": self.stats,
                    "version": "1.0",
                    "last_updated": datetime.now().isoformat()
                }
                
                with open(archive_path, 'w', encoding='utf-8') as f:
                    json.dump(archive_data, f, indent=2)
                
                # Save index
                with open(index_path, 'w', encoding='utf-8') as f:
                    json.dump(self.dream_index, f, indent=2)
                
                # Reset new dreams counter
                self.new_dreams_count = 0
                
                logger.info(f"Saved {len(self.dreams)} dreams to archive")
                
            except Exception as e:
                logger.error(f"Error saving dream archive: {e}")
    
    def _create_backup(self, file_path: Path):
        """
        Create backup of a file
        
        Args:
            file_path: Path to the file to backup
        """
        try:
            # Generate backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            backup_path = file_path.with_suffix(f".{timestamp}.backup")
            
            # Copy file to backup
            import shutil
            shutil.copy2(file_path, backup_path)
            
            # Clean up old backups if needed
            self._cleanup_backups(file_path)
            
            logger.info(f"Created backup: {backup_path}")
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
    
    def _cleanup_backups(self, original_file: Path):
        """
        Clean up old backup files
        
        Args:
            original_file: Path to the original file
        """
        try:
            # Find all backup files
            backup_pattern = f"{original_file.stem}.*{original_file.suffix}.backup"
            backup_files = list(self.data_dir.glob(backup_pattern))
            
            # Sort by modification time (oldest first)
            backup_files.sort(key=lambda p: p.stat().st_mtime)
            
            # Remove oldest backups if too many
            while len(backup_files) > self.config["max_backups"]:
                oldest = backup_files.pop(0)
                oldest.unlink()
                logger.info(f"Removed old backup: {oldest}")
                
        except Exception as e:
            logger.error(f"Error cleaning up backups: {e}")
    
    def _rebuild_index(self):
        """Rebuild the dream index from dreams list"""
        self.dream_index = {}
        
        for dream in self.dreams:
            dream_id = dream.get("id")
            if dream_id:
                # Basic index information
                self.dream_index[dream_id] = {
                    "start_time": dream.get("start_time"),
                    "duration": dream.get("metrics", {}).get("actual_duration_minutes"),
                    "intensity": dream.get("intensity"),
                    "insight_count": len(dream.get("insights", [])),
                    "phases": [p.get("phase") for p in dream.get("phases", [])]
                }
                
                # Add classification if enabled
                if self.config["dream_classification"]:
                    self.dream_index[dream_id]["classification"] = self._classify_dream(dream)
        
        logger.info(f"Rebuilt dream index with {len(self.dream_index)} entries")
    
    def _classify_dream(self, dream: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a dream based on its content
        
        Args:
            dream: Dream record to classify
            
        Returns:
            Dict with classification information
        """
        # Default classification
        classification = {
            "primary_type": "unknown",
            "secondary_type": "unknown",
            "mood": "neutral",
            "complexity": 0.5,
            "coherence": 0.5,
            "topics": []
        }
        
        try:
            # Extract insights
            insights = dream.get("insights", [])
            
            # Extract synthesis results
            synthesis_results = []
            for results in dream.get("synthesis_results", []):
                if isinstance(results, dict) and "patterns" in results:
                    synthesis_results.extend(results["patterns"])
            
            # Determine primary type based on synthesis results
            if synthesis_results:
                pattern_types = [p.get("type") for p in synthesis_results if isinstance(p, dict)]
                type_counts = {}
                for t in pattern_types:
                    if t:
                        type_counts[t] = type_counts.get(t, 0) + 1
                
                # Get most common type
                if type_counts:
                    primary_type = max(type_counts.items(), key=lambda x: x[1])[0]
                    classification["primary_type"] = primary_type
                    
                    # Get second most common type
                    if len(type_counts) > 1:
                        type_counts.pop(primary_type)
                        secondary_type = max(type_counts.items(), key=lambda x: x[1])[0]
                        classification["secondary_type"] = secondary_type
            
            # Extract topics
            topics = set()
            
            # From insights
            for insight in insights:
                text = insight.get("text", "")
                # Simple extraction of quoted or capitalized words as potential topics
                import re
                quoted = re.findall(r'["\'](.*?)["\']', text)
                capitalized = re.findall(r'\b([A-Z][a-z]+)\b', text)
                
                topics.update([t.lower() for t in quoted + capitalized if len(t) > 3])
            
            # From synthesis results
            for pattern in synthesis_results:
                if isinstance(pattern, dict):
                    # Add domains as topics
                    if "domains" in pattern:
                        topics.update(pattern["domains"])
                    
                    # Add concepts as topics
                    if "concepts" in pattern:
                        topics.update(pattern["concepts"])
            
            # Set topics
            classification["topics"] = list(topics)[:10]  # Limit to 10 topics
            
            # Calculate complexity based on insights and patterns
            insight_count = len(insights)
            pattern_count = len(synthesis_results)
            
            complexity = min(1.0, (insight_count * 0.1 + pattern_count * 0.05))
            classification["complexity"] = complexity
            
            # Set coherence
            if pattern_count > 0:
                # Average confidence of patterns
                confidences = [p.get("confidence", 0.5) for p in synthesis_results 
                              if isinstance(p, dict)]
                if confidences:
                    coherence = sum(confidences) / len(confidences)
                    classification["coherence"] = coherence
            
            # Determine mood based on insights and patterns
            if insights:
                # Simple keyword-based mood detection
                mood_keywords = {
                    "positive": ["harmony", "integration", "solution", "clarity", "insight"],
                    "negative": ["contradiction", "conflict", "problem", "confusion"],
                    "neutral": ["connection", "relation", "pattern", "structure"],
                    "creative": ["novel", "new", "creative", "innovative", "synthesis"],
                    "analytical": ["analysis", "principle", "framework", "theory", "method"]
                }
                
                mood_scores = {mood: 0 for mood in mood_keywords}
                
                # Count keywords in insights
                for insight in insights:
                    text = insight.get("text", "").lower()
                    for mood, keywords in mood_keywords.items():
                        for keyword in keywords:
                            if keyword in text:
                                mood_scores[mood] += 1
                
                # Select mood with highest score
                if any(mood_scores.values()):
                    highest_mood = max(mood_scores.items(), key=lambda x: x[1])[0]
                    classification["mood"] = highest_mood
            
            return classification
            
        except Exception as e:
            logger.error(f"Error classifying dream: {e}")
            return classification
    
    def _recalculate_stats(self):
        """Recalculate global statistics based on all dreams"""
        # Reset stats
        stats = {
            "total_dreams": len(self.dreams),
            "dreams_by_month": {},
            "most_common_topics": {},
            "insight_count": 0,
            "average_dream_duration": 0,
            "last_updated": datetime.now().isoformat()
        }
        
        # Skip if no dreams
        if not self.dreams:
            self.stats = stats
            return
        
        # Process all dreams
        durations = []
        all_topics = []
        for dream in self.dreams:
            # Count insights
            stats["insight_count"] += len(dream.get("insights", []))
            
            # Calculate durations
            if "metrics" in dream and "actual_duration_minutes" in dream["metrics"]:
                durations.append(dream["metrics"]["actual_duration_minutes"])
            
            # Group by month
            if "start_time" in dream:
                try:
                    dt = datetime.fromisoformat(dream["start_time"])
                    month_key = dt.strftime("%Y-%m")
                    stats["dreams_by_month"][month_key] = stats["dreams_by_month"].get(month_key, 0) + 1
                except (ValueError, TypeError):
                    pass
            
            # Collect topics
            classification = self._get_dream_classification(dream)
            if classification:
                all_topics.extend(classification.get("topics", []))
        
        # Calculate average duration
        if durations:
            stats["average_dream_duration"] = sum(durations) / len(durations)
        
        # Calculate most common topics
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Get top 10 topics
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        stats["most_common_topics"] = dict(sorted_topics[:10])
        
        # Update stats
        self.stats = stats
    
    def _get_dream_classification(self, dream: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get classification for a dream
        
        Args:
            dream: Dream record
            
        Returns:
            Classification dict or None
        """
        # Check if we already have classification in the index
        dream_id = dream.get("id")
        if dream_id and dream_id in self.dream_index:
            return self.dream_index[dream_id].get("classification")
        
        # Otherwise classify the dream
        if self.config["dream_classification"]:
            return self._classify_dream(dream)
        
        return None
    
    def store_dream(self, dream: Dict[str, Any]) -> bool:
        """
        Store a dream in the archive
        
        Args:
            dream: Dream record to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate dream record
            if not isinstance(dream, dict) or "id" not in dream:
                logger.error("Invalid dream record")
                return False
            
            dream_id = dream["id"]
            
            # Check if dream already exists
            existing_index = next((i for i, d in enumerate(self.dreams) if d.get("id") == dream_id), None)
            
            if existing_index is not None:
                # Update existing dream
                self.dreams[existing_index] = dream
                logger.info(f"Updated existing dream: {dream_id}")
            else:
                # Add new dream
                self.dreams.append(dream)
                logger.info(f"Added new dream: {dream_id}")
                
                # Check if we need to remove oldest dreams
                if self.config["auto_cleanup"] and len(self.dreams) > self.max_size:
                    # Sort by start_time
                    self.dreams.sort(key=lambda d: d.get("start_time", ""))
                    
                    # Remove oldest
                    removed = self.dreams.pop(0)
                    logger.info(f"Removed oldest dream: {removed.get('id', 'unknown')}")
            
            # Update index
            self.dream_index[dream_id] = {
                "start_time": dream.get("start_time"),
                "duration": dream.get("metrics", {}).get("actual_duration_minutes"),
                "intensity": dream.get("intensity"),
                "insight_count": len(dream.get("insights", [])),
                "phases": [p.get("phase") for p in dream.get("phases", [])]
            }
            
            # Add classification if enabled
            if self.config["dream_classification"]:
                self.dream_index[dream_id]["classification"] = self._classify_dream(dream)
            
            # Increment new dreams count
            self.new_dreams_count += 1
            
            # Save if needed
            if self.new_dreams_count >= self.config["persist_frequency"]:
                self._save_archive()
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing dream: {e}")
            return False
    
    def get_dream(self, dream_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific dream by ID
        
        Args:
            dream_id: ID of the dream to retrieve
            
        Returns:
            Dream record if found, None otherwise
        """
        for dream in self.dreams:
            if dream.get("id") == dream_id:
                return dream
        
        return None
    
    def get_all_dreams(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get all dreams in the archive
        
        Args:
            limit: Maximum number of dreams to return
            offset: Starting offset
            
        Returns:
            List of dream records
        """
        # Sort by start_time (newest first)
        sorted_dreams = sorted(
            self.dreams, 
            key=lambda d: d.get("start_time", ""),
            reverse=True
        )
        
        # Apply pagination
        paginated = sorted_dreams[offset:offset+limit]
        
        return paginated
    
    def search_dreams(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search dreams based on criteria
        
        Args:
            criteria: Dict with search criteria
            
        Returns:
            List of matching dream records
        """
        results = []
        
        # Initialize filters
        start_date = criteria.get("start_date")
        end_date = criteria.get("end_date")
        min_duration = criteria.get("min_duration")
        max_duration = criteria.get("max_duration")
        min_intensity = criteria.get("min_intensity")
        max_intensity = criteria.get("max_intensity")
        min_insights = criteria.get("min_insights")
        max_insights = criteria.get("max_insights")
        topics = criteria.get("topics", [])
        dream_type = criteria.get("dream_type")
        mood = criteria.get("mood")
        
        # Convert date strings to datetime if needed
        if start_date and isinstance(start_date, str):
            try:
                start_date = datetime.fromisoformat(start_date)
            except ValueError:
                start_date = None
        
        if end_date and isinstance(end_date, str):
            try:
                end_date = datetime.fromisoformat(end_date)
            except ValueError:
                end_date = None
        
        # Filter dreams
        for dream in self.dreams:
            match = True
            
            # Check start date
            if start_date and "start_time" in dream:
                try:
                    dream_date = datetime.fromisoformat(dream["start_time"])
                    if dream_date < start_date:
                        match = False
                except (ValueError, TypeError):
                    match = False
            
            # Check end date
            if match and end_date and "start_time" in dream:
                try:
                    dream_date = datetime.fromisoformat(dream["start_time"])
                    if dream_date > end_date:
                        match = False
                except (ValueError, TypeError):
                    match = False
            
            # Check duration
            if match and min_duration is not None:
                duration = dream.get("metrics", {}).get("actual_duration_minutes")
                if duration is None or duration < min_duration:
                    match = False
            
            if match and max_duration is not None:
                duration = dream.get("metrics", {}).get("actual_duration_minutes")
                if duration is not None and duration > max_duration:
                    match = False
            
            # Check intensity
            if match and min_intensity is not None:
                intensity = dream.get("intensity")
                if intensity is None or intensity < min_intensity:
                    match = False
            
            if match and max_intensity is not None:
                intensity = dream.get("intensity")
                if intensity is not None and intensity > max_intensity:
                    match = False
            
            # Check insights
            if match and min_insights is not None:
                insight_count = len(dream.get("insights", []))
                if insight_count < min_insights:
                    match = False
            
            if match and max_insights is not None:
                insight_count = len(dream.get("insights", []))
                if insight_count > max_insights:
                    match = False
            
            # Check classification criteria
            if match and (topics or dream_type or mood):
                classification = self._get_dream_classification(dream)
                
                # Check topics
                if match and topics and classification:
                    dream_topics = classification.get("topics", [])
                    if not any(topic in dream_topics for topic in topics):
                        match = False
                
                # Check dream type
                if match and dream_type and classification:
                    primary_type = classification.get("primary_type")
                    secondary_type = classification.get("secondary_type")
                    if dream_type != primary_type and dream_type != secondary_type:
                        match = False
                
                # Check mood
                if match and mood and classification:
                    dream_mood = classification.get("mood")
                    if mood != dream_mood:
                        match = False
            
            # Add to results if match
            if match:
                results.append(dream)
        
        return results
    
    def get_dream_insights(self, dream_id: str) -> List[Dict[str, Any]]:
        """
        Get insights from a specific dream
        
        Args:
            dream_id: ID of the dream
            
        Returns:
            List of insights
        """
        dream = self.get_dream(dream_id)
        if dream:
            return dream.get("insights", [])
        
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get archive statistics
        
        Returns:
            Dict with archive statistics
        """
        # Ensure stats are up to date
        self._recalculate_stats()
        return self.stats.copy()
    
    def save(self) -> bool:
        """
        Force save archive
        
        Returns:
            True if save was successful, False otherwise
        """
        try:
            self._save_archive(force=True)
            return True
        except Exception as e:
            logger.error(f"Error saving archive: {e}")
            return False
    
    def get_monthly_summary(self) -> Dict[str, Any]:
        """
        Get summary of dreams by month
        
        Returns:
            Dict with monthly statistics
        """
        # Ensure stats are up to date
        if not self.stats.get("dreams_by_month"):
            self._recalculate_stats()
        
        monthly_data = self.stats.get("dreams_by_month", {})
        
        # Sort months
        sorted_months = sorted(monthly_data.items())
        
        # Calculate stats per month
        monthly_stats = {}
        
        for month, count in sorted_months:
            # Get dreams for this month
            month_dreams = []
            for dream in self.dreams:
                if "start_time" in dream:
                    try:
                        dream_date = datetime.fromisoformat(dream["start_time"])
                        if dream_date.strftime("%Y-%m") == month:
                            month_dreams.append(dream)
                    except (ValueError, TypeError):
                        pass
            
            # Skip if no dreams found
            if not month_dreams:
                continue
            
            # Calculate stats
            insights = sum(len(dream.get("insights", [])) for dream in month_dreams)
            
            durations = [
                dream.get("metrics", {}).get("actual_duration_minutes") 
                for dream in month_dreams
                if dream.get("metrics", {}).get("actual_duration_minutes") is not None
            ]
            
            avg_duration = sum(durations) / len(durations) if durations else 0
            
            # Get most common topics this month
            all_topics = []
            for dream in month_dreams:
                classification = self._get_dream_classification(dream)
                if classification:
                    all_topics.extend(classification.get("topics", []))
            
            topic_counts = {}
            for topic in all_topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
            # Get top 5 topics
            sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
            top_topics = dict(sorted_topics[:5]) if sorted_topics else {}
            
            # Store monthly stats
            monthly_stats[month] = {
                "count": count,
                "insight_count": insights,
                "avg_duration": avg_duration,
                "top_topics": top_topics
            }
        
        return monthly_stats 