# Lumina v8 CI/CD Pipeline Documentation

This document describes the Continuous Integration and Continuous Deployment (CI/CD) pipeline for the Lumina v8 system.

## Overview

The CI/CD pipeline automates the process of testing, validating, and deploying the Lumina v8 system components. It ensures that all components are functioning correctly and properly integrated before deployment.

## Pipeline Stages

The pipeline consists of the following stages:

1. **Environment Validation**
   - Checks the Python environment and version
   - Ensures required directories exist
   - Validates system setup
   - Verifies Mistral API key availability

2. **Dependency Validation**
   - Checks for required Python packages
   - Automatically installs missing dependencies
   - Validates installation success

3. **Hardware & Capability Check**
   - Tests GPU capabilities
   - Validates Qt3D modules availability
   - Checks OpenGL support

4. **Component Testing**
   - Runs unit tests for each component
   - Validates component integration
   - Ensures all core functionality works

5. **Execution**
   - Offers multiple execution options
   - Records logs for all stages
   - Reports any issues or failures

## Usage

To run the CI/CD pipeline, execute the `run_complete_knowledge_cycle.bat` file:

```
.\run_complete_knowledge_cycle.bat
```

This will start the pipeline and guide you through the options.

## Pipeline Options

After validation, the pipeline offers several options:

1. **Start Integrated Mode** - Runs all components together in an integrated environment
2. **Start Root Connection Only** - Runs only the root connection system
3. **Start Visualization Mode** - Launches multiple components with visualization
4. **Run Tests Only** - Executes tests without starting any components
5. **Start Lumina v7.5 Chat Interface** - Launches the interactive chat system

## Integrated Chat Interface (Lumina v7.5)

The pipeline now includes the Lumina v7.5 Chat Interface, which provides:

- Interactive communication with the Lumina system
- Real-time monitoring of system components and status
- Integration with the CI/CD pipeline
- Conversation memory that tracks topics across exchanges
- Support for both connected and mock modes

The chat interface can be used to:
- Query the status of pipeline components
- Receive updates about system processes
- Interact with the neural network capabilities
- Explore CI/CD concepts through natural language

The chat system will attempt to use a Mistral API key if available, looking in:
1. Environment variables (`MISTRAL_API_KEY`)
2. The `.env` file in the project root

If no key is found, the system will run in mock mode with simulated responses.

## Log Files

The pipeline generates detailed logs in the `logs` directory with timestamps. Test results are stored in the `test_results` directory.

Chat interface logs are stored in `logs/v7.5_frontend.log`.

## Troubleshooting

If the pipeline fails at any stage:

1. Check the log file for detailed error messages
2. Verify that all dependencies are properly installed
3. Ensure compatible hardware for 3D visualization
4. Validate the Python environment

For visualization issues, try running in 2D mode using the `--mode 2d` flag with the Spatial Temple component.

For chat interface issues, check the API key availability and PySide6 installation.

## Extending the Pipeline

To add new components or tests to the pipeline:

1. Create a new test file in the `src/tests` directory
2. Add the test class to the test suite in `run_all_tests.py`
3. Update the requirements.txt file if needed
4. Modify the batch file to include new component validation 