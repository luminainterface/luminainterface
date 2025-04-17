@echo off
setlocal enabledelayedexpansion

echo.
echo ===================================
echo LUMINA V7.5 AutoWiki Standalone
echo ===================================
echo.

REM Create required directories if they don't exist
if not exist "data" mkdir data
if not exist "logs" mkdir logs
if not exist "data\autowiki" mkdir data\autowiki
if not exist "logs\autowiki" mkdir logs\autowiki
if not exist "data\knowledge" mkdir data\knowledge
if not exist "data\cache" mkdir data\cache

REM Load API key from .env file if it exists
if exist .env (
    echo Loading API key from .env file...
    FOR /F "tokens=2 delims==" %%a in ('type .env ^| findstr "MISTRAL_API_KEY"') do (
        set MISTRAL_API_KEY=%%a
    )
) else (
    REM Fallback to hardcoded key if no .env file
    set MISTRAL_API_KEY=nLKZEpq29OihnaArxV7s6KtzsNEiky2A
)

    echo Using Mistral API Key: %MISTRAL_API_KEY:~0,4%...%MISTRAL_API_KEY:~-4%

REM Set environment variables
set PYTHONPATH=%CD%
set LUMINA_HOME=%CD%
set LUMINA_AUTOWIKI_PORT=7525
set ENABLE_AUTOWIKI=true
set ENABLE_AUTO_FETCH=true
set AUTO_FETCH_INTERVAL=60
set KNOWLEDGE_DIR=data\knowledge
set CACHE_DIR=data\cache
set AUTOWIKI_DB_PATH=data\knowledge\wiki_db.sqlite
set SHARED_DB_ENABLED=true
set SHARED_DB_PATH=data\neural_metrics.db
set CONNECTION_STRING=sqlite:///data/neural_metrics.db?check_same_thread=False

REM Check and install required packages
echo Checking and installing required packages...
python -m pip install --upgrade pip

REM List of required packages
set PACKAGES=mistralai requests beautifulsoup4 feedparser pyside6 markdown nltk

REM Install each package if not already installed
for %%p in (%PACKAGES%) do (
    python -c "import %%p" 2>NUL
    if !ERRORLEVEL! NEQ 0 (
        echo Installing %%p...
        python -m pip install %%p
    ) else (
        echo %%p already installed.
    )
)

REM Create standalone autowiki module if needed
echo Checking for AutoWiki implementation...
set FOUND_AUTOWIKI=false

REM Try v7_5 first (most likely to have the autowiki module)
if exist src\v7_5\autowiki.py (
    echo Found autowiki.py in v7_5 directory
    set FOUND_AUTOWIKI=true
    set AUTOWIKI_PATH=src\v7_5\autowiki.py
    goto :launch_autowiki
)

REM Try in v7.5 directory
if exist src\v7.5 (
    echo Checking v7.5 directory for wiki modules...
for %%f in (src\v7.5\*wiki*.py) do (
        echo Found %%f
        set FOUND_AUTOWIKI=true
        set AUTOWIKI_PATH=%%f
        goto :launch_autowiki
    )
)

REM Try other potential locations
if exist src\autowiki (
    echo Checking src\autowiki directory...
    for %%f in (src\autowiki\*.py) do (
        echo Found %%f
        set FOUND_AUTOWIKI=true
        set AUTOWIKI_PATH=%%f
        goto :launch_autowiki
    )
)

REM Check if any file has autowiki in the name
for %%f in (src\*autowiki*.py src\v7\*wiki*.py) do (
    echo Found %%f
    set FOUND_AUTOWIKI=true
    set AUTOWIKI_PATH=%%f
    goto :launch_autowiki
)

REM If no AutoWiki found, create a standalone implementation
if "%FOUND_AUTOWIKI%"=="false" (
    echo No existing AutoWiki module found.
    echo Creating standalone AutoWiki implementation...
    
    mkdir standalone_autowiki 2>NUL
    
    echo import os, sys, time, json, threading, random, datetime > standalone_autowiki\autowiki.py
    echo import requests >> standalone_autowiki\autowiki.py
    echo try: >> standalone_autowiki\autowiki.py
    echo     import sqlite3 >> standalone_autowiki\autowiki.py
    echo except ImportError: >> standalone_autowiki\autowiki.py
    echo     print("Warning: sqlite3 not available. Database features will be limited.") >> standalone_autowiki\autowiki.py
    echo. >> standalone_autowiki\autowiki.py
    echo class StandaloneAutoWiki: >> standalone_autowiki\autowiki.py
    echo     def __init__(self, api_key=None, port=7525, data_dir="data/autowiki", knowledge_dir="data/knowledge", cache_dir="data/cache", db_path=None): >> standalone_autowiki\autowiki.py
    echo         self.api_key = api_key or os.environ.get("MISTRAL_API_KEY") >> standalone_autowiki\autowiki.py
    echo         self.port = port >> standalone_autowiki\autowiki.py
    echo         self.data_dir = data_dir >> standalone_autowiki\autowiki.py
    echo         self.knowledge_dir = knowledge_dir >> standalone_autowiki\autowiki.py
    echo         self.cache_dir = cache_dir >> standalone_autowiki\autowiki.py
    echo         self.db_path = db_path or os.environ.get("AUTOWIKI_DB_PATH") or os.path.join(data_dir, "wiki_db.sqlite") >> standalone_autowiki\autowiki.py
    echo         self.shared_db = os.environ.get("SHARED_DB_ENABLED", "false").lower() in ["true", "1", "yes"] >> standalone_autowiki\autowiki.py
    echo         self.shared_db_path = os.environ.get("SHARED_DB_PATH") >> standalone_autowiki\autowiki.py
    echo         self.running = False >> standalone_autowiki\autowiki.py
    echo         self.topics = [] >> standalone_autowiki\autowiki.py
    echo         self.fetch_thread = None >> standalone_autowiki\autowiki.py
    echo         self.auto_fetch = True >> standalone_autowiki\autowiki.py
    echo         self.auto_fetch_interval = int(os.environ.get("AUTO_FETCH_INTERVAL", 60)) >> standalone_autowiki\autowiki.py
    echo         self.setup() >> standalone_autowiki\autowiki.py
    echo. >> standalone_autowiki\autowiki.py
    echo     def setup(self): >> standalone_autowiki\autowiki.py
    echo         print(f"Setting up AutoWiki at port {self.port}") >> standalone_autowiki\autowiki.py
    echo         os.makedirs(self.data_dir, exist_ok=True) >> standalone_autowiki\autowiki.py
    echo         os.makedirs(self.knowledge_dir, exist_ok=True) >> standalone_autowiki\autowiki.py
    echo         os.makedirs(self.cache_dir, exist_ok=True) >> standalone_autowiki\autowiki.py
    echo         self.load_topics() >> standalone_autowiki\autowiki.py
    echo         self.init_db() >> standalone_autowiki\autowiki.py
    echo. >> standalone_autowiki\autowiki.py
    echo     def init_db(self): >> standalone_autowiki\autowiki.py
    echo         try: >> standalone_autowiki\autowiki.py
    echo             print(f"Initializing database: {self.db_path}") >> standalone_autowiki\autowiki.py
    echo             conn = sqlite3.connect(self.db_path) >> standalone_autowiki\autowiki.py
    echo             cursor = conn.cursor() >> standalone_autowiki\autowiki.py
    echo             cursor.execute('''CREATE TABLE IF NOT EXISTS wiki_articles >> standalone_autowiki\autowiki.py
    echo                           (topic TEXT PRIMARY KEY, content TEXT, last_updated TIMESTAMP, file_path TEXT)''') >> standalone_autowiki\autowiki.py
    echo             conn.commit() >> standalone_autowiki\autowiki.py
    echo             conn.close() >> standalone_autowiki\autowiki.py
    echo             print("Database initialized successfully") >> standalone_autowiki\autowiki.py
    echo. >> standalone_autowiki\autowiki.py
    echo             # Check shared database connection if enabled >> standalone_autowiki\autowiki.py
    echo             if self.shared_db and self.shared_db_path: >> standalone_autowiki\autowiki.py
    echo                 print(f"Checking shared database: {self.shared_db_path}") >> standalone_autowiki\autowiki.py
    echo                 if os.path.exists(self.shared_db_path): >> standalone_autowiki\autowiki.py
    echo                     try: >> standalone_autowiki\autowiki.py
    echo                         shared_conn = sqlite3.connect(self.shared_db_path) >> standalone_autowiki\autowiki.py
    echo                         shared_cursor = shared_conn.cursor() >> standalone_autowiki\autowiki.py
    echo                         shared_cursor.execute('''CREATE TABLE IF NOT EXISTS autowiki_status >> standalone_autowiki\autowiki.py
    echo                                           (id INTEGER PRIMARY KEY, status TEXT, last_update TIMESTAMP)''') >> standalone_autowiki\autowiki.py
    echo                         shared_conn.commit() >> standalone_autowiki\autowiki.py
    echo                         print("Connected to shared database successfully") >> standalone_autowiki\autowiki.py
    echo                         # Add status entry >> standalone_autowiki\autowiki.py
    echo                         shared_cursor.execute("INSERT OR REPLACE INTO autowiki_status (id, status, last_update) VALUES (1, ?, ?)", >> standalone_autowiki\autowiki.py
    echo                                              ("online", datetime.datetime.now().isoformat())) >> standalone_autowiki\autowiki.py
    echo                         shared_conn.commit() >> standalone_autowiki\autowiki.py
    echo                         shared_conn.close() >> standalone_autowiki\autowiki.py
    echo                     except Exception as e: >> standalone_autowiki\autowiki.py
    echo                         print(f"Error connecting to shared database: {e}") >> standalone_autowiki\autowiki.py
    echo                 else: >> standalone_autowiki\autowiki.py
    echo                     print(f"Shared database file not found: {self.shared_db_path}") >> standalone_autowiki\autowiki.py
    echo         except Exception as e: >> standalone_autowiki\autowiki.py
    echo             print(f"Error initializing database: {e}") >> standalone_autowiki\autowiki.py
    echo. >> standalone_autowiki\autowiki.py
    echo     def load_topics(self): >> standalone_autowiki\autowiki.py
    echo         try: >> standalone_autowiki\autowiki.py
    echo             topics_file = os.path.join(self.data_dir, "topics.json") >> standalone_autowiki\autowiki.py
    echo             if os.path.exists(topics_file): >> standalone_autowiki\autowiki.py
    echo                 with open(topics_file, "r") as f: >> standalone_autowiki\autowiki.py
    echo                     self.topics = json.load(f) >> standalone_autowiki\autowiki.py
    echo                 print(f"Loaded {len(self.topics)} topics from {topics_file}") >> standalone_autowiki\autowiki.py
    echo             else: >> standalone_autowiki\autowiki.py
    echo                 print("No existing topics file. Creating default topics.") >> standalone_autowiki\autowiki.py
    echo                 self.topics = ["artificial intelligence", "neural networks", "machine learning", >> standalone_autowiki\autowiki.py
    echo                               "large language models", "consciousness", "cognitive science"] >> standalone_autowiki\autowiki.py
    echo                 self.save_topics() >> standalone_autowiki\autowiki.py
    echo         except Exception as e: >> standalone_autowiki\autowiki.py
    echo             print(f"Error loading topics: {e}") >> standalone_autowiki\autowiki.py
    echo             self.topics = ["artificial intelligence", "neural networks"] >> standalone_autowiki\autowiki.py
    echo. >> standalone_autowiki\autowiki.py
    echo     def save_topics(self): >> standalone_autowiki\autowiki.py
    echo         try: >> standalone_autowiki\autowiki.py
    echo             topics_file = os.path.join(self.data_dir, "topics.json") >> standalone_autowiki\autowiki.py
    echo             with open(topics_file, "w") as f: >> standalone_autowiki\autowiki.py
    echo                 json.dump(self.topics, f) >> standalone_autowiki\autowiki.py
    echo             print(f"Saved {len(self.topics)} topics to {topics_file}") >> standalone_autowiki\autowiki.py
    echo         except Exception as e: >> standalone_autowiki\autowiki.py
    echo             print(f"Error saving topics: {e}") >> standalone_autowiki\autowiki.py
    echo. >> standalone_autowiki\autowiki.py
    echo     def fetch_content(self, topic): >> standalone_autowiki\autowiki.py
    echo         try: >> standalone_autowiki\autowiki.py
    echo             print(f"Fetching content for: {topic}") >> standalone_autowiki\autowiki.py
    echo             # Basic method to fetch content using API or simulation >> standalone_autowiki\autowiki.py
    echo             if self.api_key: >> standalone_autowiki\autowiki.py
    echo                 try: >> standalone_autowiki\autowiki.py
    echo                     from mistralai.client import MistralClient >> standalone_autowiki\autowiki.py
    echo                     from mistralai.models.chat_completion import ChatMessage >> standalone_autowiki\autowiki.py
    echo                     client = MistralClient(api_key=self.api_key) >> standalone_autowiki\autowiki.py
    echo                     messages = [ >> standalone_autowiki\autowiki.py
    echo                         ChatMessage(role="system", content=f"You are a helpful assistant that provides factual information about {topic}."), >> standalone_autowiki\autowiki.py
    echo                         ChatMessage(role="user", content=f"Write a comprehensive but concise encyclopedia entry about {topic}. Include the most important facts, history, and current understanding.") >> standalone_autowiki\autowiki.py
    echo                     ] >> standalone_autowiki\autowiki.py
    echo                     chat_response = client.chat(model="mistral-medium", messages=messages) >> standalone_autowiki\autowiki.py
    echo                     content = chat_response.choices[0].message.content >> standalone_autowiki\autowiki.py
    echo                     print(f"Received content from Mistral API: {len(content)} characters") >> standalone_autowiki\autowiki.py
    echo                 except Exception as e: >> standalone_autowiki\autowiki.py
    echo                     print(f"Error using Mistral API: {e}") >> standalone_autowiki\autowiki.py
    echo                     content = f"# {topic.title()}\n\nInformation about {topic} is currently being researched.\n\nLast updated: {datetime.datetime.now().strftime('%%Y-%%m-%%d %%H:%%M:%%S')}" >> standalone_autowiki\autowiki.py
    echo             else: >> standalone_autowiki\autowiki.py
    echo                 content = f"# {topic.title()}\n\nInformation about {topic} is currently being researched.\n\nLast updated: {datetime.datetime.now().strftime('%%Y-%%m-%%d %%H:%%M:%%S')}" >> standalone_autowiki\autowiki.py
    echo. >> standalone_autowiki\autowiki.py
    echo             # Save the content >> standalone_autowiki\autowiki.py
    echo             filename = topic.replace(" ", "_").replace("/", "_").lower() + ".md" >> standalone_autowiki\autowiki.py
    echo             filepath = os.path.join(self.knowledge_dir, filename) >> standalone_autowiki\autowiki.py
    echo             with open(filepath, "w", encoding="utf-8") as f: >> standalone_autowiki\autowiki.py
    echo                 f.write(content) >> standalone_autowiki\autowiki.py
    echo             print(f"Saved content for {topic} to {filepath}") >> standalone_autowiki\autowiki.py
    echo. >> standalone_autowiki\autowiki.py
    echo             # Store in database >> standalone_autowiki\autowiki.py
    echo             try: >> standalone_autowiki\autowiki.py
    echo                 conn = sqlite3.connect(self.db_path) >> standalone_autowiki\autowiki.py
    echo                 cursor = conn.cursor() >> standalone_autowiki\autowiki.py
    echo                 now = datetime.datetime.now().isoformat() >> standalone_autowiki\autowiki.py
    echo                 cursor.execute("INSERT OR REPLACE INTO wiki_articles (topic, content, last_updated, file_path) VALUES (?, ?, ?, ?)", >> standalone_autowiki\autowiki.py
    echo                               (topic, content, now, filepath)) >> standalone_autowiki\autowiki.py
    echo                 conn.commit() >> standalone_autowiki\autowiki.py
    echo                 conn.close() >> standalone_autowiki\autowiki.py
    echo                 print(f"Stored content in database for topic: {topic}") >> standalone_autowiki\autowiki.py
    echo. >> standalone_autowiki\autowiki.py
    echo                 # Update shared database if enabled >> standalone_autowiki\autowiki.py
    echo                 if self.shared_db and self.shared_db_path and os.path.exists(self.shared_db_path): >> standalone_autowiki\autowiki.py
    echo                     try: >> standalone_autowiki\autowiki.py
    echo                         shared_conn = sqlite3.connect(self.shared_db_path) >> standalone_autowiki\autowiki.py
    echo                         shared_cursor = shared_conn.cursor() >> standalone_autowiki\autowiki.py
    echo                         # Update status with latest activity >> standalone_autowiki\autowiki.py
    echo                         shared_cursor.execute("INSERT OR REPLACE INTO autowiki_status (id, status, last_update) VALUES (1, ?, ?)", >> standalone_autowiki\autowiki.py
    echo                                            (f"Updated topic: {topic}", now)) >> standalone_autowiki\autowiki.py
    echo                         # Create content log entry >> standalone_autowiki\autowiki.py
    echo                         shared_cursor.execute('''CREATE TABLE IF NOT EXISTS autowiki_content_log >> standalone_autowiki\autowiki.py
    echo                                            (id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT, timestamp TIMESTAMP, 
    echo                                             file_path TEXT, character_count INTEGER)''') >> standalone_autowiki\autowiki.py
    echo                         shared_cursor.execute("INSERT INTO autowiki_content_log (topic, timestamp, file_path, character_count) VALUES (?, ?, ?, ?)", >> standalone_autowiki\autowiki.py
    echo                                            (topic, now, filepath, len(content))) >> standalone_autowiki\autowiki.py
    echo                         shared_conn.commit() >> standalone_autowiki\autowiki.py
    echo                         shared_conn.close() >> standalone_autowiki\autowiki.py
    echo                         print(f"Updated shared database with content info for topic: {topic}") >> standalone_autowiki\autowiki.py
    echo                     except Exception as e: >> standalone_autowiki\autowiki.py
    echo                         print(f"Error updating shared database: {e}") >> standalone_autowiki\autowiki.py
    echo             except Exception as e: >> standalone_autowiki\autowiki.py
    echo                 print(f"Error storing content in database: {e}") >> standalone_autowiki\autowiki.py
    echo. >> standalone_autowiki\autowiki.py
    echo             return content >> standalone_autowiki\autowiki.py
    echo         except Exception as e: >> standalone_autowiki\autowiki.py
    echo             print(f"Error fetching content for {topic}: {e}") >> standalone_autowiki\autowiki.py
    echo             return None >> standalone_autowiki\autowiki.py
    echo. >> standalone_autowiki\autowiki.py
    echo     def auto_fetch_loop(self): >> standalone_autowiki\autowiki.py
    echo         print(f"Starting auto-fetch thread with interval {self.auto_fetch_interval} seconds") >> standalone_autowiki\autowiki.py
    echo         while self.running and self.auto_fetch: >> standalone_autowiki\autowiki.py
    echo             try: >> standalone_autowiki\autowiki.py
    echo                 if self.topics: >> standalone_autowiki\autowiki.py
    echo                     topic = random.choice(self.topics) >> standalone_autowiki\autowiki.py
    echo                     self.fetch_content(topic) >> standalone_autowiki\autowiki.py
    echo             except Exception as e: >> standalone_autowiki\autowiki.py
    echo                 print(f"Error in auto fetch: {e}") >> standalone_autowiki\autowiki.py
    echo             time.sleep(self.auto_fetch_interval) >> standalone_autowiki\autowiki.py
    echo. >> standalone_autowiki\autowiki.py
    echo     def start(self): >> standalone_autowiki\autowiki.py
    echo         print("Starting AutoWiki...") >> standalone_autowiki\autowiki.py
    echo         self.running = True >> standalone_autowiki\autowiki.py
    echo         if self.auto_fetch: >> standalone_autowiki\autowiki.py
    echo             self.fetch_thread = threading.Thread(target=self.auto_fetch_loop) >> standalone_autowiki\autowiki.py
    echo             self.fetch_thread.daemon = True >> standalone_autowiki\autowiki.py
    echo             self.fetch_thread.start() >> standalone_autowiki\autowiki.py
    echo         print("AutoWiki running in standalone mode") >> standalone_autowiki\autowiki.py
    echo         print(f"Content is being saved to {self.knowledge_dir}") >> standalone_autowiki\autowiki.py
    echo         try: >> standalone_autowiki\autowiki.py
    echo             while self.running: >> standalone_autowiki\autowiki.py
    echo                 time.sleep(1) >> standalone_autowiki\autowiki.py
    echo         except KeyboardInterrupt: >> standalone_autowiki\autowiki.py
    echo             print("\nShutting down AutoWiki...") >> standalone_autowiki\autowiki.py
    echo             self.running = False >> standalone_autowiki\autowiki.py
    echo             if self.fetch_thread: >> standalone_autowiki\autowiki.py
    echo                 self.fetch_thread.join(timeout=1) >> standalone_autowiki\autowiki.py
    echo. >> standalone_autowiki\autowiki.py
    echo. >> standalone_autowiki\autowiki.py
    echo if __name__ == "__main__": >> standalone_autowiki\autowiki.py
    echo     import argparse >> standalone_autowiki\autowiki.py
    echo     parser = argparse.ArgumentParser(description="Standalone AutoWiki") >> standalone_autowiki\autowiki.py
    echo     parser.add_argument("--port", type=int, default=7525, help="Port to run on") >> standalone_autowiki\autowiki.py
    echo     parser.add_argument("--data-dir", default="data/autowiki", help="Data directory") >> standalone_autowiki\autowiki.py
    echo     parser.add_argument("--knowledge-dir", default="data/knowledge", help="Knowledge directory") >> standalone_autowiki\autowiki.py
    echo     parser.add_argument("--no-auto-fetch", action="store_true", help="Disable auto fetching") >> standalone_autowiki\autowiki.py
    echo     parser.add_argument("--db-path", help="Path to database file") >> standalone_autowiki\autowiki.py
    echo     parser.add_argument("--shared-db", action="store_true", help="Enable shared database") >> standalone_autowiki\autowiki.py
    echo     parser.add_argument("--shared-db-path", help="Path to shared database file") >> standalone_autowiki\autowiki.py
    echo     args = parser.parse_args() >> standalone_autowiki\autowiki.py
    echo     wiki = StandaloneAutoWiki( >> standalone_autowiki\autowiki.py
    echo         port=args.port, >> standalone_autowiki\autowiki.py
    echo         data_dir=args.data_dir, >> standalone_autowiki\autowiki.py
    echo         knowledge_dir=args.knowledge_dir, >> standalone_autowiki\autowiki.py
    echo         db_path=args.db_path >> standalone_autowiki\autowiki.py
    echo     ) >> standalone_autowiki\autowiki.py
    echo     wiki.auto_fetch = not args.no_auto_fetch >> standalone_autowiki\autowiki.py
    echo     if args.shared_db: >> standalone_autowiki\autowiki.py
    echo         wiki.shared_db = True >> standalone_autowiki\autowiki.py
    echo     if args.shared_db_path: >> standalone_autowiki\autowiki.py
    echo         wiki.shared_db_path = args.shared_db_path >> standalone_autowiki\autowiki.py
    echo     wiki.start() >> standalone_autowiki\autowiki.py
    
    set FOUND_AUTOWIKI=true
    set AUTOWIKI_PATH=standalone_autowiki\autowiki.py
)

:launch_autowiki
echo.
echo Using AutoWiki module: %AUTOWIKI_PATH%
echo.

REM Create a Windows environment variable for the API key so the script can access it
set MISTRAL_API_KEY=%MISTRAL_API_KEY%

REM Launch the selected AutoWiki module
echo All systems ready. Starting AutoWiki...
python %AUTOWIKI_PATH% --port=%LUMINA_AUTOWIKI_PORT% --data-dir=data\autowiki --knowledge-dir=%KNOWLEDGE_DIR%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: AutoWiki exited with code %ERRORLEVEL%
    echo Check if the arguments are correct for this version of AutoWiki.
    echo Trying with simplified arguments...
    
    REM Try with less arguments in case the module doesn't support all options
    python %AUTOWIKI_PATH% --port=%LUMINA_AUTOWIKI_PORT%
    
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo Error: AutoWiki still failed with simplified arguments.
        echo Trying with no arguments...
        
        python %AUTOWIKI_PATH%
    )
)

echo.
echo AutoWiki session completed.
    pause