import os
import sys
from dotenv import load_dotenv
from ui.app import main as run_app

def setup_environment():
    """Check and setup the environment before running the app."""
    # Load environment variables
    load_dotenv()
    
    # Check required environment variables
    required_vars = ['PINECONE_API_KEY', 'ROBOFLOW_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"- {var}")
        sys.exit(1)
    
    # Create required directories
    os.makedirs("data/processed/text", exist_ok=True)
    os.makedirs("data/processed/images", exist_ok=True)

if __name__ == "__main__":
    setup_environment()
    run_app() 