import os
import glob
import json
import concurrent.futures
from utils import setup_logger, get_openai_client, call_llm_with_retry, save_json, load_json, validate_json_list, BufferedLogger
import config

# ================= Configuration =================
MODEL_NAME = config.GENERATION_MODEL
CAPTION_DIR = config.CAPTION_DIR
OUTPUT_DIR = "results/step1_qa_generation"
START_INDEX = 0
END_INDEX = 467
MAX_WORKERS = config.MAX_WORKERS_DEFAULT
# =============================================

logger = setup_logger("Step1_Generation", OUTPUT_DIR)
client = get_openai_client()

def load_captions_text(file_path):
    data = load_json(file_path)
    if not data: return ""
    
    context_str = ""
    for clip in data:
        slice_id = clip.get('slice_num') or clip.get('slice_id')
        caption = clip.get('cap') or clip.get('caption')
        if slice_id is not None:
            context_str += f"[Slice_{slice_id}]: {caption}\n\n"
    return context_str

def get_system_prompt():
    return """
# Role
You are an expert architect of Video Understanding Benchmarks. Your goal is to create a high-quality "Multi-Hop Video QA Dataset" for **Long-Context Video Understanding**.

# Input Data Schema
The input provided is a raw text log of visual captions formatted as:
`[Slice_ID]: {Caption Text}`
* **Slice_ID**: An integer representing the chronological order.
* **Caption**: Visual description of that specific segment.
* **Note**: Slice N is immediately followed by Slice N+1.

# Data Context
The input comes from a **SINGLE CONTINUOUS LONG VIDEO**.
1.  **Continuity:** The video flows linearly. Events are causally linked across time.
2.  **VLM Noise:** Entities may have inconsistent names. Link them via **visual attributes** (e.g., specific clothing, objects) rather than names.

# CRITICAL: Anti-Noise Protocol (Abstract Logic)
You must rigorously filter out "Pseudo Multi-hop" questions:
* ❌ **Pseudo Multi-hop (INVALID):** Reasoning relies entirely on information within a **single slice**.
    * *Test:* `evidence_slices` contains only one unique integer (e.g., `[A, A]`). -> **REJECT.**
* ❌ **Duplicate Slices (INVALID):** Do not list the same slice ID multiple times in `evidence_slices`.
    * *Test:* `evidence_slices` must NOT contain duplicates (e.g., `[A, A, B]` is INVALID). If multiple clues are in Slice A, list A only once.
* ✅ **True Multi-hop (REQUIRED):** The question connects information from **at least two distinct time segments**.
    * *Test:* `evidence_slices` must contain **unique integers** (e.g., `[A, B]` where `A != B`).

# Hop Level Definitions (Strictly 2/3/4 Only)
You must generate questions with specific complexity levels:
* **2-Hop:** Combines info from exactly 2 distinct slices.
* **3-Hop:** Connects 3 slices.
* **4-Hop:** Connects 4 distinct slices.
* **Constraint:** Do NOT generate 1-hop or 5+ hop questions.
* **Distribution Requirement:** You MUST strictly ensure that the number of **2-Hop, 3-Hop, and 4-Hop questions are approximately equal (Ratio 1:1:1)**.

# Question Categories
1.  **State_Mutation:** Comparing an entity's state at **Time T1** vs **Time T2**.
2.  **Causal_Inference:** The **Cause** is at **Time T1**, and the **Effect** is at **Time T2**.
3.  **Visual_Tracking:** Identifying the same entity across distinct scenes based on visual features.
4.  **Global_Summary:** Aggregating information across the entire video timeline.

# Task
Generate **at least 20 distinct multi-hop questions** (the more, the better).

# Constraints
1.  **Quantity:** **Maximize the number of questions.** You must output **at least 20**.
2.  **Distinct Indices:** `evidence_slices` MUST strictly contain at least two distinct numbers.
3.  **Global Distribution:** Ensure the questions are distributed across the **entire video timeline**.
4.  **Natural Language:** Do not refer to "Slice IDs" in the question text.
5.  **Self-Contained:** The question must be understandable without context.

# Output Format
Return **ONLY** a raw JSON list.
[
  {
    "question": "string",
    "answer": "string",
    "category": "State_Mutation", 
    "hop_level": "2-Hop",
    "evidence_slices": [12, 45], 
    "reasoning_chain": "Step 1: Slice 12 shows... Step 2: Slice 45 shows... Conclusion..."
  }
]
"""

def extract_valid_json_objects(text):
    """Robust parser for potentially truncated JSON streams."""
    if not text: return []
    clean_text = text.replace("```json", "").replace("```", "").strip()
    
    objects = []
    stack = 0
    start_idx = -1
    
    try:
        for i, char in enumerate(clean_text):
            if char == '{':
                if stack == 0: start_idx = i
                stack += 1
            elif char == '}':
                stack -= 1
                if stack == 0 and start_idx != -1:
                    json_str = clean_text[start_idx : i+1]
                    try:
                        obj = json.loads(json_str)
                        if "question" in obj: objects.append(obj)
                    except json.JSONDecodeError: pass
                    start_idx = -1
        return objects
    except Exception as e:
        logger.error(f"Parsing error: {e}")
        return []

def process_single_video(input_file):
    video_id = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(OUTPUT_DIR, f"{video_id}_multihop_qa.json")
    
    # Use BufferedLogger to prevent interleaved logs
    local_logger = BufferedLogger(logger, prefix=f"[{video_id}] ")
    
    local_logger.info(f"🎬 Processing Video: {video_id}")
    
    if os.path.exists(output_file):
        local_logger.info(f"⏩ File exists, skipping.")
        local_logger.flush()
        return

    context_text = load_captions_text(input_file)
    if not context_text: 
        local_logger.flush()
        return

    user_prompt = f"""
# Video Context (Temporal Log)
{context_text}

# Generation Instructions
Please generate **at least 20 multi-hop questions** based on the context above.
**Goal:** Exhaustively mine the video for all valid multi-hop connections.
**Ratio:** Remember to keep the ratio of 2-Hop, 3-Hop, and 4-Hop questions roughly **1:1:1**.
"""
    
    local_logger.info(f"🤖 Generating questions with {MODEL_NAME}...")
    raw_response = call_llm_with_retry(
        client, MODEL_NAME, 
        [{"role": "system", "content": get_system_prompt()}, {"role": "user", "content": user_prompt}],
        temperature=0.7, max_tokens=8192, logger=local_logger,
        validator=validate_json_list
    )

    if raw_response:
        parsed_json = extract_valid_json_objects(raw_response)
        if parsed_json:
            counts = {}
            for q in parsed_json:
                h = q.get('hop_level', 'Other')
                counts[h] = counts.get(h, 0) + 1
            
            save_json(parsed_json, output_file)
            local_logger.info(f"✅ Success! Extracted {len(parsed_json)} questions. Dist: {counts}")
        else:
            local_logger.error("❌ Parsed list is empty.")
            with open(os.path.join(OUTPUT_DIR, f"{video_id}_error_raw.txt"), "w", encoding='utf-8') as f:
                f.write(raw_response)
    else:
        local_logger.error(f"❌ Failed to generate valid JSON for {video_id} after retries.")
        with open(os.path.join(OUTPUT_DIR, "failed_videos.log"), "a", encoding='utf-8') as f:
            f.write(f"{video_id}\n")
            
    local_logger.flush()

if __name__ == "__main__":
    if not os.path.exists(CAPTION_DIR):
        logger.error(f"Directory not found: {CAPTION_DIR}")
    else:
        all_files = sorted(glob.glob(os.path.join(CAPTION_DIR, "*.json")))
        target_files = all_files[START_INDEX:END_INDEX]
        logger.info(f"🎯 Processing indices {START_INDEX}-{END_INDEX} (Total {len(target_files)}) with {MAX_WORKERS} threads")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            executor.map(process_single_video, target_files)
