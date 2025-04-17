"""
Article Manager for AutoWiki System
Handles database operations and article management
"""

import os
import json
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ArticleManager:
    def __init__(self, db_path: str = "data/wiki/articles.db"):
        self.db_path = db_path
        self._ensure_db_path()
        self.conn = None
        self.cursor = None
        self._initialize_db()
        
    def _ensure_db_path(self):
        """Ensure database directory exists"""
        db_dir = os.path.dirname(self.db_path)
        os.makedirs(db_dir, exist_ok=True)
        
    def _initialize_db(self):
        """Initialize database and create tables if they don't exist"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            
            # Create articles table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Create revisions table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS revisions (
                    id TEXT PRIMARY KEY,
                    article_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    author TEXT,
                    metadata TEXT,
                    FOREIGN KEY (article_id) REFERENCES articles (id)
                )
            """)
            
            self.conn.commit()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
            
    def create_article(self, title: str, content: str, category: str = None, metadata: Dict = None) -> str:
        """Create a new article"""
        try:
            article_id = f"article_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            now = datetime.now().isoformat()
            
            self.cursor.execute("""
                INSERT INTO articles (id, title, content, category, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                article_id,
                title,
                content,
                category,
                now,
                now,
                json.dumps(metadata or {})
            ))
            
            self.conn.commit()
            logger.info(f"Created article: {title} ({article_id})")
            return article_id
            
        except Exception as e:
            logger.error(f"Failed to create article: {str(e)}")
            raise
            
    def update_article(self, article_id: str, content: str, author: str = None, metadata: Dict = None) -> bool:
        """Update an existing article and create revision"""
        try:
            # Create revision
            revision_id = f"rev_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            now = datetime.now().isoformat()
            
            self.cursor.execute("""
                INSERT INTO revisions (id, article_id, content, created_at, author, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                revision_id,
                article_id,
                content,
                now,
                author,
                json.dumps(metadata or {})
            ))
            
            # Update article
            self.cursor.execute("""
                UPDATE articles 
                SET content = ?, updated_at = ?
                WHERE id = ?
            """, (content, now, article_id))
            
            self.conn.commit()
            logger.info(f"Updated article {article_id} with revision {revision_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update article: {str(e)}")
            raise
            
    def get_article(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Get article by ID"""
        try:
            self.cursor.execute("""
                SELECT id, title, content, category, created_at, updated_at, metadata
                FROM articles WHERE id = ?
            """, (article_id,))
            
            result = self.cursor.fetchone()
            if not result:
                return None
                
            return {
                'id': result[0],
                'title': result[1],
                'content': result[2],
                'category': result[3],
                'created_at': result[4],
                'updated_at': result[5],
                'metadata': json.loads(result[6])
            }
            
        except Exception as e:
            logger.error(f"Failed to get article: {str(e)}")
            raise
            
    def get_article_revisions(self, article_id: str) -> List[Dict[str, Any]]:
        """Get all revisions for an article"""
        try:
            self.cursor.execute("""
                SELECT id, content, created_at, author, metadata
                FROM revisions WHERE article_id = ?
                ORDER BY created_at DESC
            """, (article_id,))
            
            revisions = []
            for row in self.cursor.fetchall():
                revisions.append({
                    'id': row[0],
                    'content': row[1],
                    'created_at': row[2],
                    'author': row[3],
                    'metadata': json.loads(row[4])
                })
                
            return revisions
            
        except Exception as e:
            logger.error(f"Failed to get article revisions: {str(e)}")
            raise
            
    def search_articles(self, query: str, category: str = None) -> List[Dict[str, Any]]:
        """Search articles by content and optionally filter by category"""
        try:
            if category:
                self.cursor.execute("""
                    SELECT id, title, content, category, created_at, updated_at, metadata
                    FROM articles 
                    WHERE (title LIKE ? OR content LIKE ?) AND category = ?
                """, (f"%{query}%", f"%{query}%", category))
            else:
                self.cursor.execute("""
                    SELECT id, title, content, category, created_at, updated_at, metadata
                    FROM articles 
                    WHERE title LIKE ? OR content LIKE ?
                """, (f"%{query}%", f"%{query}%"))
                
            results = []
            for row in self.cursor.fetchall():
                results.append({
                    'id': row[0],
                    'title': row[1],
                    'content': row[2],
                    'category': row[3],
                    'created_at': row[4],
                    'updated_at': row[5],
                    'metadata': json.loads(row[6])
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Failed to search articles: {str(e)}")
            raise
            
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None 