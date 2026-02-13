import os
import glob
import json
import time
import base64
import concurrent.futures
from utils import setup_logger, get_openai_client, call_llm_with_retry, save_json, load_json, parse_json_from_response, validate_json_obj, BufferedLogger
import config

# ================= Configuration =================
MODEL_NAME = config.VIDEO_VERIFICATION_MODEL # VLM Model
CAPTION_DIR = config.CAPTION_DIR
VIDEO_ROOT_DIR = config.VIDEO_ROOT_DIR

INPUT_DIR = "results/step5_necessity_check"   # Input from Step 5
OUTPUT_DIR = "results/step6_video_verification" # Output for Step 6
START_INDEX = 0
END_INDEX = 467
MAX_WORKERS = config.MAX_WORKERS_VIDEO
# =============================================

logger = setup_logger("Step5_VideoVerification", OUTPUT_DIR)
client = get_openai_client()

def get_clip_path(video_id, slice_id):
    """Finds the video clip file (supports fuzzy matching)."""
    slice_str = str(slice_id).zfill(3)
    video_folder = os.path.join(VIDEO_ROOT_DIR, video_id)
    
    # Compatibility: If ID folder doesn't exist, try root
    if not os.path.exists(video_folder):
        video_folder = VIDEO_ROOT_DIR
    
    # 1. Try standard naming
    filename = f"{video_id}-Scene-{slice_str}.mp4"
    full_path = os.path.join(video_folder, filename)
    if os.path.exists(full_path):
        return full_path
    
    # 2. Try wildcard matching
    patterns = [
        os.path.join(video_folder, f"*-Scene-{slice_str}.mp4"),
        os.path.join(video_folder, f"*{slice_id}.mp4")
    ]
    for p in patterns:
        matches = glob.glob(p)
        if matches: return matches[0]
        
    return None

def encode_video_to_base64(video_path):
    """Reads video file and converts to Base64 string."""
    if not os.path.exists(video_path): return None
    try:
        with open(video_path, "rb") as video_file:
            return base64.b64encode(video_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding video {video_path}: {e}")
        return None

def verify_visual_logic(qa_item, video_id, logger=logger):
    question = qa_item['question']
    original_answer = qa_item['answer']
    evidence_ids = qa_item['evidence_slices']
    
    if not evidence_ids: return False, "No evidence slices", None
    
    messages_content = []
    clip_count = len(evidence_ids)
    
    system_instruction = f"""
You are an expert **Video QA Auditor**.
You will be provided with **{clip_count} distinct video clips** to analyze.

**🎥 EVIDENCE CONTEXT (CRITICAL):**
1. **Single Source:** All {clip_count} clips are extracted from the **SAME continuous long video**. 
2. **Temporal Discontinuity:** Although from the same source, these clips are **time-sliced**.

**🕵️ YOUR TASK:**
1. **Visual Verification:** Can the question be answered *purely* based on the visual information in these {clip_count} clips?
2. **Fact Check:** Validate the "Original Answer".
   - If the answer describes something visible in the clips, verify its accuracy.
   - If the Original Answer is **visually contradicted** (e.g., wrong color, wrong person, action didn't happen), provide a `refined_answer`.

**INPUT DATA:**
* **Question:** {question}
* **Original Answer:** {original_answer}

**OUTPUT FORMAT (JSON):**
{{
  "verdict_is_answerable": true/false,
  "unanswerable_reason": "Explanation if false...",
  "verdict_is_correct": true/false,
  "refined_answer": "Corrected answer or Original answer.",
  "visual_proof": "Briefly describe the specific visual details from the clips that support your verdict."
}}
"""
    messages_content.append({"type": "text", "text": system_instruction})

    found_video = False
    
    # --- Process each Evidence Slice ---
    for i, sid in enumerate(evidence_ids):
        v_path = get_clip_path(video_id, sid)
        if v_path:
            b64_video = encode_video_to_base64(v_path)
            if b64_video:
                found_video = True
                clip_header = f"\n\n=== 🎞️ CLIP {i+1}/{clip_count} (ID: {sid}) ==="
                messages_content.append({"type": "text", "text": clip_header})
                messages_content.append({
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{b64_video}"}
                })
        else:
            logger.warning(f"  ⚠️ Clip {sid} not found for video {video_id}.")

    if not found_video:
        return False, "No video files found", None
    
    logger.info(f"  📸 Sending {clip_count} video clips to VLM...")

    # --- API Call ---
    response_text = call_llm_with_retry(
        client, 
        MODEL_NAME, 
        [{"role": "user", "content": messages_content}], 
        temperature=0.0, 
        max_tokens=8192, 
        logger=logger,
        validator=validate_json_obj
    )
    
    if response_text:
        return True, "Success", response_text
    else:
        return False, "API Call Failed (Retries Exhausted)", None

def process_single_video(caption_file):
    video_id = os.path.splitext(os.path.basename(caption_file))[0]
    
    input_file = os.path.join(INPUT_DIR, f"{video_id}_passed_necessity_check.json")
    output_file = os.path.join(OUTPUT_DIR, f"{video_id}_passed_video_check.json")
    failed_file = os.path.join(OUTPUT_DIR, f"{video_id}_failed_video_check.json")
    
    local_logger = BufferedLogger(logger, prefix=f"[{video_id}] ")
    local_logger.info(f"🎥 Step 6 (Video Verification): {video_id}")

    if not os.path.exists(input_file):
        local_logger.warning(f"⚠️ Input file {input_file} not found, skipping.")
        local_logger.flush()
        return
        
    if os.path.exists(output_file):
        local_logger.info(f"⏩ File exists, skipping.")
        local_logger.flush()
        return

    raw_qas = load_json(input_file)
    if not raw_qas: 
        local_logger.flush()
        return

    passed_qas = []
    failed_qas = []
    
    local_logger.info(f"🚀 Verifying {len(raw_qas)} questions...")

    for idx, qa in enumerate(raw_qas):
        success, msg, resp_text = verify_visual_logic(qa, video_id, logger=local_logger)
        
        if success:
            result_json = parse_json_from_response(resp_text)
            
            if result_json:
                is_answerable = result_json.get("verdict_is_answerable", False)
                is_correct = result_json.get("verdict_is_correct", False)
                
                if is_answerable:
                    if is_correct:
                        local_logger.info(f"Q{idx+1}: ✅ Passed (Original Correct)")
                    else:
                        local_logger.info(f"Q{idx+1}: ✅ Passed (Refined)")
                        qa['original_text_answer'] = qa['answer']
                        qa['answer'] = result_json.get("refined_answer", qa['answer'])
                        qa['verdict_meta'] = "REFINED"

                    qa['visual_proof'] = result_json.get("visual_proof", "Verified by VLM")
                    passed_qas.append(qa)
                else:
                    reason = result_json.get("unanswerable_reason", "Unknown")
                    local_logger.info(f"Q{idx+1}: ❌ Failed (Unanswerable: {reason[:50]}...)")
                    qa['failure_reason'] = reason
                    failed_qas.append(qa)
            else:
                local_logger.warning(f"Q{idx+1}: ⚠️ JSON Parse Error")
                qa['failure_reason'] = "JSON Parse Error"
                failed_qas.append(qa)
        else:
            local_logger.error(f"Q{idx+1}: ⚠️ API Error: {msg}")
            qa['failure_reason'] = f"API Error: {msg}"
            failed_qas.append(qa)
        
        # Avoid rate limits
        time.sleep(1)

    save_json(passed_qas, output_file)

    if failed_qas:
        save_json(failed_qas, failed_file)
        local_logger.info(f"📉 Saved {len(failed_qas)} failed questions to: {failed_file}")

    local_logger.info(f"🎉 Done. Retention rate: {len(passed_qas)}/{len(raw_qas)}")
    local_logger.flush()

if __name__ == "__main__":
    all_files = sorted(glob.glob(os.path.join(CAPTION_DIR, "*.json")))
    target_files = all_files[START_INDEX:END_INDEX]
    
    logger.info(f"🎯 Step 5 processing indices {START_INDEX}-{END_INDEX} (Total {len(target_files)}) with {MAX_WORKERS} threads")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(process_single_video, target_files)
