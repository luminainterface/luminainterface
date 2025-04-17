@echo off
echo Starting LUMINA V7 Wiki Knowledge Builder...

:: Create necessary directories
if not exist data mkdir data
if not exist data\chat_memory mkdir data\chat_memory
if not exist data\auto_wiki mkdir data\auto_wiki
if not exist config mkdir config
if not exist logs mkdir logs
if not exist scripts mkdir scripts

:: Check if topics file exists
set TOPICS_FILE=data\wiki_topics.txt
if not exist %TOPICS_FILE% (
    echo Creating default topics file...
    echo artificial intelligence > %TOPICS_FILE%
    echo neural networks >> %TOPICS_FILE%
    echo machine learning >> %TOPICS_FILE%
    echo consciousness >> %TOPICS_FILE%
    echo cognition >> %TOPICS_FILE%
    echo linguistics >> %TOPICS_FILE%
    echo natural language processing >> %TOPICS_FILE%
    echo deep learning >> %TOPICS_FILE%
    echo reinforcement learning >> %TOPICS_FILE%
    echo language models >> %TOPICS_FILE%
)

:: Check for Mistral API key
set MISTRAL_API_KEY_FILE=config\mistral_api_key.txt
set MISTRAL_API_KEY=
if exist %MISTRAL_API_KEY_FILE% (
    for /f "tokens=*" %%a in (%MISTRAL_API_KEY_FILE%) do set MISTRAL_API_KEY=%%a
)

if "%MISTRAL_API_KEY%"=="" (
    echo No Mistral API key found. Running without Mistral integration.
    echo To enable Mistral integration, create a file at %MISTRAL_API_KEY_FILE% with your API key.
    python scripts\run_wiki_knowledge_builder.py --topics-file %TOPICS_FILE% --progress-file data\wiki_builder_progress.json
) else (
    echo Mistral API key found. Running with Mistral integration.
    python scripts\run_wiki_knowledge_builder.py --topics-file %TOPICS_FILE% --progress-file data\wiki_builder_progress.json --mistral-api-key %MISTRAL_API_KEY% --mistral-model mistral-medium
)

echo Wiki Knowledge Builder completed.
echo Results are stored in the data directory.
pause 