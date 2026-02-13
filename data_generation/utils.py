import logging
import os
import json
import time
import threading
from openai import OpenAI
import config

# ================= Common Utilities =================

# Global lock for atomic logging
LOG_LOCK = threading.Lock()

class BufferedLogger:
    """Collects logs and flushes them atomically to prevent interleaving in threaded environments."""
    def __init__(self, logger, prefix=""):
        self.logger = logger
        self.prefix = prefix
        self.buffer = []

    def info(self, msg):
        self.buffer.append((logging.INFO, msg))

    def error(self, msg):
        self.buffer.append((logging.ERROR, msg))
        
    def warning(self, msg):
        self.buffer.append((logging.WARNING, msg))

    def flush(self):
        with LOG_LOCK:
            for level, msg in self.buffer:
                full_msg = f"{self.prefix}{msg}"
                self.logger.log(level, full_msg)
            self.buffer = []

def get_openai_client():
    if not config.API_KEY:
        # Fallback only for testing locally if env variable not set but user edits file directly
        # raise ValueError("API Key is missing. Please set OPEN_MODEL_API_KEY environment variable.")
        pass
    return OpenAI(api_key=config.API_KEY, base_url=config.API_BASE_URL)


def setup_logger(name, log_dir):
    """Sets up a logger that writes to both file and console."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, f"{name.lower()}.log")
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate logs if logger is already set up
    if logger.handlers:
        logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setFormatter(formatter)
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def load_json(file_path):
    """Safely loads a JSON file."""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"❌ Error loading JSON {file_path}: {e}")
        return None

def save_json(data, file_path):
    """Safely saves data to a JSON file."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logging.error(f"❌ Error saving JSON {file_path}: {e}")
        return False

def call_llm_with_retry(client, model, messages, temperature=0.1, max_tokens=4096, max_retries=3, logger=None, validator=None):
    """Executes an LLM API call with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=180 # 3 minutes timeout
            )
            content = response.choices[0].message.content
            
            if validator:
                if validator(content):
                    return content
                else:
                    msg = f"Validation failed for attempt {attempt + 1}"
                    if logger: logger.warning(f"⚠️ {msg}")
                    # Treat as exception to trigger retry logic
                    raise ValueError(msg)
            
            return content
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ API Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                if logger:
                    logger.error(f"❌ API Final Failure: {e}")
                return None

def parse_json_from_response(text):
    """Extracts and parses JSON object from LLM response text."""
    if not text: return None
    try:
        # Try direct parse
        return json.loads(text)
    except json.JSONDecodeError:
        # Try stripping markdown
        clean_text = text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError:
            return None

def validate_json_list(text):
    """Checks if the response contains a valid JSON list."""
    if not text: return False
    try:
        # Try to find the first '[' and last ']'
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            json.loads(candidate)
            return True
        return False
    except:
        return False

def validate_json_obj(text):
    """Checks if the response contains a valid JSON object."""
    if not text: return False
    try:
        # Try to find the first '{' and last '}'
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            json.loads(candidate)
            return True
        return False
    except:
        return False

def load_captions_map(file_path):
    """Loads a caption file and returns a {slice_num: caption} dictionary."""
    data = load_json(file_path)
    if not data: return {}
    return {item.get('slice_num'): item.get('cap') for item in data}
