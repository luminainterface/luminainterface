from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import redis
import json
import os
from typing import List, Optional, Dict, Set
import logging
from prometheus_client import Counter, Histogram
import time
import httpx
import asyncio
from bs4 import BeautifulSoup
import aiohttp
import git
import tempfile
import shutil
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(title="Crawler Service")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Redis client
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(redis_url)

# Prometheus metrics
crawl_requests = Counter('crawl_requests_total', 'Total number of crawl requests')
crawl_latency = Histogram('crawl_latency_seconds', 'Time spent crawling')
pages_crawled = Counter('pages_crawled_total', 'Total number of pages crawled')
git_events = Counter('git_events_total', 'Total git events processed', ['event_type'])
git_files_processed = Counter('git_files_processed_total', 'Total git files processed', ['status'])

# File extensions to process
ALLOWED_EXTENSIONS = {
    '.md', '.txt', '.rst', '.py', '.js', '.ts', '.java', '.cpp', '.h', 
    '.hpp', '.c', '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt'
}

# Directories to skip
SKIP_DIRS = {
    '.git', 'node_modules', 'venv', '__pycache__', 'target', 'build',
    'dist', 'vendor', 'bower_components'
}

class CrawlRequest(BaseModel):
    url: str
    depth: Optional[int] = 1
    max_pages: Optional[int] = 10

class CrawlResponse(BaseModel):
    url: str
    title: str
    content: str
    links: List[str]

class GitWebhook(BaseModel):
    repository: Dict
    commits: List[Dict]
    ref: str

class GitFile(BaseModel):
    path: str
    content: str
    sha: str
    license: Optional[str] = None

@app.get("/health")
async def health_check():
    try:
        redis_client.ping()
        return {"status": "healthy", "redis": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

async def fetch_page(url: str) -> Optional[Dict]:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract title
                    title = soup.title.string if soup.title else ""
                    
                    # Extract main content (simplified)
                    content = soup.get_text(separator=' ', strip=True)
                    
                    # Extract links
                    links = [a.get('href') for a in soup.find_all('a', href=True)]
                    
                    return {
                        "url": url,
                        "title": title,
                        "content": content,
                        "links": links
                    }
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return None

@app.post("/crawl")
async def crawl(request: CrawlRequest):
    with crawl_latency.time():
        crawl_requests.inc()
        try:
            # Fetch the initial page
            page_data = await fetch_page(request.url)
            if not page_data:
                raise HTTPException(status_code=404, detail="Failed to fetch page")
            
            # Store in Redis
            key = f"crawl:{request.url}"
            redis_client.setex(key, 3600, json.dumps(page_data))  # 1 hour TTL
            
            pages_crawled.inc()
            return CrawlResponse(**page_data)
            
        except Exception as e:
            logger.error(f"Error crawling {request.url}: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

async def process_git_file(file_path: Path, repo_path: Path) -> Optional[GitFile]:
    """Process a single git file and extract its content."""
    try:
        if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            git_files_processed.labels(status="skipped_extension").inc()
            return None
            
        # Skip files in excluded directories
        if any(skip_dir in file_path.parts for skip_dir in SKIP_DIRS):
            git_files_processed.labels(status="skipped_directory").inc()
            return None
            
        full_path = repo_path / file_path
        if not full_path.exists():
            git_files_processed.labels(status="not_found").inc()
            return None
            
        # Read file content
        content = full_path.read_text(encoding='utf-8', errors='ignore')
        
        # Get file SHA
        repo = git.Repo(repo_path)
        sha = repo.git.rev_parse(f'HEAD:{file_path}')
        
        # Try to detect license
        license_file = repo_path / 'LICENSE'
        license = None
        if license_file.exists():
            license = license_file.read_text(encoding='utf-8', errors='ignore')[:100]
        
        git_files_processed.labels(status="success").inc()
        return GitFile(
            path=str(file_path),
            content=content,
            sha=sha,
            license=license
        )
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        git_files_processed.labels(status="error").inc()
        return None

async def clone_and_process_repo(repo_url: str, branch: str = "main") -> List[GitFile]:
    """Clone a repository and process its files."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Clone repository
        repo = git.Repo.clone_from(repo_url, temp_dir, branch=branch)
        repo_path = Path(temp_dir)
        
        # Process all files
        processed_files = []
        for file_path in repo_path.rglob("*"):
            if file_path.is_file():
                if result := await process_git_file(file_path.relative_to(repo_path), repo_path):
                    processed_files.append(result)
        
        return processed_files
    finally:
        shutil.rmtree(temp_dir)

@app.post("/webhook/git")
async def git_webhook(request: Request):
    """Handle git webhook events."""
    try:
        payload = await request.json()
        event_type = request.headers.get('X-GitHub-Event', 'push')
        git_events.labels(event_type=event_type).inc()
        
        if event_type == 'push':
            webhook_data = GitWebhook(**payload)
            repo_url = webhook_data.repository['clone_url']
            branch = webhook_data.ref.split('/')[-1]
            
            # Process repository
            files = await clone_and_process_repo(repo_url, branch)
            
            # Queue files for ingestion
            for file in files:
                payload = {
                    "type": "gitfile",
                    "payload": {
                        "path": file.path,
                        "content": file.content,
                        "sha": file.sha,
                        "license": file.license,
                        "repo": repo_url,
                        "branch": branch
                    }
                }
                redis_client.publish("ingest.raw_html", json.dumps(payload))
            
            return {"status": "success", "files_processed": len(files)}
        else:
            return {"status": "ignored", "event_type": event_type}
            
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn