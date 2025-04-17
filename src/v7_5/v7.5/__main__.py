"""
LUMINA v7.5 Main Entry Point

This module allows running the LUMINA frontend directly with:
python -m src.v7.5
"""

import sys
import os
from . import parse_args
from .lumina_frontend import main

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Set environment variables based on arguments
    if args.interlink:
        os.environ['V7_5_INTERLINK'] = 'true'
    if args.port:
        os.environ['V7_5_PORT'] = str(args.port)
    if args.log_level:
        os.environ['LOG_LEVEL'] = args.log_level
    
    # Start the appropriate interface
    sys.exit(main(args)) 