import sys
import os

# Add the src directory to the Python path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.append(src_path)

# Import and run the main application
from frontend.main import main

if __name__ == '__main__':
    main() 