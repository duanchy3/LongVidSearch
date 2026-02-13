import os
import glob
import itertools
import time
import concurrent.futures
from utils import setup_logger, get_openai_client, call_llm_with_retry, save_json, load_json, parse_json_from_response, validate_json_obj, load_captions_map, BufferedLogger
import config

# ================= Configuration =================
MODEL_NAME = config.NECESSITY_CHECK_MODEL
CAPTION_DIR = config.CAPTION_DIR
INPUT_DIR = "results/step4_logic_check"     # Input from Step 4
OUTPUT_DIR = "results/step5_necessity_check" # Output for Step 5
START_INDEX = 0
END_INDEX = 467
MAX_WORKERS = config.MAX_WORKERS_DEFAULT
# =============================================

logger = setup_logger("Step4_NecessityCheck", OUTPUT_DIR)
client = get_openai_client()

def run_necessity_check(question, evidence_ids, caption_map):
    """
    Test B (Necessity Check): 
    Ensure that removing any single piece of evidence makes the question unanswerable.
    If a subset (N-1) is sufficient to answer, then the removed piece was not necessary.
    """
    evidence_texts = [caption_map.get(sid, "") for sid in evidence_ids]
    total_evidence_count = len(evidence_texts)
    
    # Generate all N-1 combinations
    subsets_indices = list(itertools.combinations(range(total_evidence_count), total_evidence_count - 1))
    
    if not subsets_indices:
        return False, "Evidence count < 2", ""

    for subset_idx in subsets_indices:
        # Build "Incomplete" Context
        current_subset_texts = [evidence_texts[i] for i in subset_idx]
        current_subset_ids = [evidence_ids[i] for i in subset_idx]
        
        subset_context = ""
        for i, text in enumerate(current_subset_texts):
            subset_context += f"[Fragment {evidence_ids[subset_idx[i]]}]: {text}\n"

        prompt = f"""
### Incomplete Context (I have DELETED one crucial evidence slice)
{subset_context}

### Question
{question}

### Task: The "Missing Link" Test
The original question required {total_evidence_count} pieces of evidence. I have **removed one**.
Act as a **Literal Robot** with NO common sense.

**Can you strictly deduce the full answer using ONLY the text above?**

### CRITICAL INSTRUCTIONS (Handling Naming Inconsistency)
1.  **Awareness:** Be aware that entities in this dataset may be described with **inconsistent names or attributes** across different fragments.
2.  **The Strict Identity Rule:** Since a linking fragment is missing, you are **STRICTLY FORBIDDEN** from assuming that two differently named entities (e.g., "Entity A" and "Entity B") are the same object/character.
    * *Logic:* Without the intermediate context to explain the transformation or connection, you must treat them as separate, unrelated entities.
    * *Action:* If the answer relies on linking these ambiguous entities, you MUST return "INSUFFICIENT".
3.  **No Hallucination:** Do not infer causal links that are not explicitly written in the remaining text.

### Output Format (Strict JSON)
Return ONLY a raw JSON object.
{{
    "missing_analysis": "Abstractly analyze what logical link or identity definition is missing...",
    "verdict": "SOLVABLE" (if fully answerable) or "INSUFFICIENT" (if info is missing)
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
            verdict = result_json.get("verdict", "INSUFFICIENT").upper()
            analysis = result_json.get("missing_analysis", "")
            
            # If ANY subset is solvable, the test fails (because not all pieces were necessary)
            if verdict == "SOLVABLE":
                return False, f"Fail: Solvable by subset {current_subset_ids}. Reasoning: {analysis}", analysis
        else:
            # Fallback text match
            if response_text and "SOLVABLE" in response_text.upper():
                 return False, f"Fail: Solvable (Text Match) by subset {current_subset_ids}", "Text Match"

    return True, "Passed Strict N-1 Test", ""

def process_single_video(caption_file):
    video_id = os.path.splitext(os.path.basename(caption_file))[0]
    
    input_file = os.path.join(INPUT_DIR, f"{video_id}_passed_logic_check.json")
    output_file = os.path.join(OUTPUT_DIR, f"{video_id}_passed_necessity_check.json")
    failed_file = os.path.join(OUTPUT_DIR, f"{video_id}_failed_necessity_check.json")
    
    local_logger = BufferedLogger(logger, prefix=f"[{video_id}] ")
    local_logger.info(f"🛡️ Step 5 (Necessity Check): {video_id}")

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
        evidence_ids = qa.get('evidence_slices', [])
        is_valid, reason, missing_analysis = run_necessity_check(qa['question'], evidence_ids, local_caption_map)
        
        if is_valid:
            local_logger.info(f"Q{idx+1}: ✅ Passed")
            passed_qas.append(qa)
        else:
            local_logger.info(f"Q{idx+1}: ❌ Failed ({reason[:50]}...)")
            qa['failure_reason'] = reason
            qa['missing_analysis'] = missing_analysis
            failed_qas.append(qa)
        
        # Small delay to avoid rate limits if running very fast, though retry handles it.
        # time.sleep(0.2) 

    save_json(passed_qas, output_file)

    if failed_qas:
        save_json(failed_qas, failed_file)
        local_logger.info(f"📉 Saved {len(failed_qas)} failed questions to: {failed_file}")

    local_logger.info(f"🎉 Done. Retention rate: {len(passed_qas)}/{len(raw_qas)}")
    local_logger.flush()

if __name__ == "__main__":
    all_files = sorted(glob.glob(os.path.join(CAPTION_DIR, "*.json")))
    target_files = all_files[START_INDEX:END_INDEX]
    
    logger.info(f"🎯 Step 4 processing indices {START_INDEX}-{END_INDEX} (Total {len(target_files)}) with {MAX_WORKERS} threads")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(process_single_video, target_files)
