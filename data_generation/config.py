import os

# ================= Global Configuration =================

# API Configuration
# Users should set these environment variables before running the pipeline
API_KEY = os.getenv("OPEN_MODEL_API_KEY", "") 
API_BASE_URL = os.getenv("OPEN_MODEL_API_BASE", "https://api.openai.com/v1") 

# Data Paths
# Adjust these paths to match your local environment
# Defaulting to relative paths or placeholders for open source release

# Path to the directory containing video caption JSON files
CAPTION_DIR = os.getenv("Start_CAPTION_DIR", "/path/to/single_video_captions")

# Path to the directory containing the actual video files (for Step 6)
VIDEO_ROOT_DIR = os.getenv("Start_VIDEO_ROOT_DIR", "/path/to/video_data/long_video_clip")

# Model Names
# Adjust model names based on your deployment
GENERATION_MODEL = "gpt-5.2"
LEAKAGE_CHECK_MODEL = "gpt-5"
LOGIC_CHECK_MODEL = "gpt-5"
NECESSITY_CHECK_MODEL = "gpt-5"
VIDEO_VERIFICATION_MODEL = "qwen3-vl-235b-a22b-instruct"

# Concurrency
MAX_WORKERS_DEFAULT = 30
MAX_WORKERS_VIDEO = 8

def validate_config():
    """Validates that necessary configuration is set."""
    missing = []
    if not API_KEY:
        missing.append("OPEN_MODEL_API_KEY")
    
    # We warn but don't fail for paths, as the user might not run all steps
    if not os.path.exists(CAPTION_DIR):
        print(f"Warning: CAPTION_DIR {CAPTION_DIR} does not exist.")
        
    if not os.path.exists(VIDEO_ROOT_DIR):
        print(f"Warning: VIDEO_ROOT_DIR {VIDEO_ROOT_DIR} does not exist.")

    if missing:
        print("Warning: The following environment variables are missing:")
        for m in missing:
            print(f"  - {m}")
        print("API calls will likely fail.")

