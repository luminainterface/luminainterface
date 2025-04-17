# Lumina Frontend

A PySide6-based frontend application for monitoring and interacting with Mistral LLM.

## Features

- Unified launcher for accessing different components
- Real-time system monitoring (CPU, Memory, GPU)
- Chat interface with Mistral LLM integration
- Adjustable LLM parameters (temperature, top-k, top-p)

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Launch the application:
   ```bash
   python main.py
   ```
2. Use the launcher to open either the monitoring screen or chat interface
3. In the chat interface, adjust parameters using the sliders or spin boxes
4. Type messages in the input field and press Enter or click Send

## Development

### Code Style

- Use Black for code formatting
- Use Ruff for linting
- Follow PEP 8 guidelines

### Testing

Run tests using pytest:
```bash
pytest tests/
```

## CI/CD

The project includes GitHub Actions workflows for:
- Automated testing
- Code formatting and linting
- Build and deployment

## Database

The application uses SQLite for local storage of:
- Chat history
- System metrics
- User preferences

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License 