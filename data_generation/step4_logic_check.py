import os
import glob
import json
import concurrent.futures
from utils import setup_logger, get_openai_client, call_llm_with_retry, save_json, load_json, parse_json_from_response, validate_json_obj, load_captions_map, BufferedLogger
import config

# ================= Configuration =================
MODEL_NAME = config.LOGIC_CHECK_MODEL
CAPTION_DIR = config.CAPTION_DIR
INPUT_DIR = "results/step3_clean"
OUTPUT_DIR = "results/step4_logic_check"  # Output for Step 4
START_INDEX = 0
END_INDEX = 467
MAX_WORKERS = config.MAX_WORKERS_DEFAULT
# =============================================

logger = setup_logger("Step3_LogicCheck", OUTPUT_DIR)
client = get_openai_client()

def run_logic_check(question, answer, evidence_ids, caption_map):
    evidence_texts = [caption_map.get(sid, "") for sid in evidence_ids]
    if any(not t for t in evidence_texts): 
        return False, "Some Slice IDs not found in Caption file", ""

    context_str = ""
    for i, text in enumerate(evidence_texts):
        context_str += f"[Evidence Slice {evidence_ids[i]}]: {text}\n"

    prompt = f"""
### Context (Visual Captions from a Video)
{context_str}

### Question
{question}

### Proposed Answer
{answer}

### Role
You are a strict QA Verifier. Your job is to check if the **Context** fully supports the **Answer**.

### CRITICAL INSTRUCTIONS
1. **Handle VLM Noise:** These captions are AI-generated. The same character might be named differently in different slices (e.g., "a man in blue" vs "the driver"). **Do NOT fail** just because of naming mismatches if the visual attributes (clothes, actions) align logically.
2. **Chain of Thought:** You must trace the logic step-by-step. Does Slice X link to Slice Y logically?
3. **Factuality:** Does the text explicitly support the answer? Do not allow external knowledge.

### Output Format (Strict JSON)
Return ONLY a raw JSON object. No markdown.
{{
    "reasoning": "Step 1: Slice A says... Step 2: Slice B says... Logic holds because...",
    "verdict": "PASS" or "FAIL"
}}
"""
    response_text = call_llm_with_retry(
        client, MODEL_NAME, 
        [{"role": "user", "content": prompt}], 
        temperature=0.1, logger=logger,
        validator=validate_json_obj
    )
    
    if not response_text:
        return False, "API Call Failed (Retries Exhausted)", "API Error"

    result_json = parse_json_from_response(response_text)
    
    if result_json:
        verdict = result_json.get("verdict", "FAIL").upper()
        reason = result_json.get("reasoning", "No reasoning provided")
        
        if verdict == "PASS":
            return True, "Passed", reason
        else:
            return False, f"LLM Rejected: {reason}", reason
    else:
        # Fallback text match (should rarely happen with validator)
        if response_text and "PASS" in response_text.upper():
            return True, "Passed (Text Match)", "Text Match (No JSON)"
        return False, f"Parse Error or Fail: {response_text}", "Parse Error"

def process_single_video(caption_file):
    video_id = os.path.splitext(os.path.basename(caption_file))[0]
    
    input_file = os.path.join(INPUT_DIR, f"{video_id}_deduplicated.json")
    output_file = os.path.join(OUTPUT_DIR, f"{video_id}_passed_logic_check.json")
    failed_file = os.path.join(OUTPUT_DIR, f"{video_id}_failed_logic_check.json")
    
    local_logger = BufferedLogger(logger, prefix=f"[{video_id}] ")
    local_logger.info(f"🧪 Step 4 (Logic Check): {video_id}")

    if not os.path.exists(input_file):
        local_logger.warning(f"⚠️ Input file {input_file} not found, skipping.")
        local_logger.flush()
        return
        
    if os.path.exists(output_file):
        local_logger.info(f"⏩ File exists, skipping.")
        local_logger.flush()
        return

    local_caption_map = load_captions_map(caption_file)
    if not local_caption_map: 
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
        # Ensure cleanup of old fields
        qa.pop('test_a_reasoning', None)
        
        evidence_ids = qa.get('evidence_slices', [])
        if len(set(evidence_ids)) < 2:
            qa['failure_reason'] = "Not enough evidence slices (<2)"
            failed_qas.append(qa)
            continue
            
        is_valid, msg, reasoning = run_logic_check(qa['question'], qa['answer'], evidence_ids, local_caption_map)
        
        if is_valid:
            local_logger.info(f"Q{idx+1}: ✅ Passed")
            qa['logic_check_reasoning'] = reasoning
            passed_qas.append(qa)
        else:
            local_logger.info(f"Q{idx+1}: ❌ Failed ({msg[:50]}...)") 
            qa['failure_reason'] = msg
            failed_qas.append(qa)

    save_json(passed_qas, output_file)

    if failed_qas:
        save_json(failed_qas, failed_file)
        local_logger.info(f"📉 Saved {len(failed_qas)} failed questions to: {failed_file}")

    local_logger.info(f"🎉 Done. Retention rate: {len(passed_qas)}/{len(raw_qas)}")
    local_logger.flush()

if __name__ == "__main__":
    all_files = sorted(glob.glob(os.path.join(CAPTION_DIR, "*.json")))
    target_files = all_files[START_INDEX:END_INDEX]
    
    logger.info(f"🎯 Step 3 processing indices {START_INDEX}-{END_INDEX} (Total {len(target_files)}) with {MAX_WORKERS} threads")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(process_single_video, target_files)
