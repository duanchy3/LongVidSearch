import os
import glob
import json
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import setup_logger, get_openai_client, call_llm_with_retry, save_json, load_json
import config

# ================= Configuration =================
INPUT_DIR = "results/step2_deduplication"
OUTPUT_DIR = "results/step3_leakage_check" # Directory for intermediate files and final results
FINAL_CLEAN_DIR = "results/step3_clean"   # Where clean files go

# Intermediate files
FULL_CONTEXT_FILE = os.path.join(OUTPUT_DIR, 'all_qs_full_context.json')
LLM_INPUT_FILE = os.path.join(OUTPUT_DIR, 'qas_for_review.json')
BAD_IDS_FILE = os.path.join(OUTPUT_DIR, 'bad_qa_ids.json')
WRONG_QA_FILE = os.path.join(OUTPUT_DIR, 'wrong_question.json')

MODEL_NAME = config.LEAKAGE_CHECK_MODEL

BATCH_SIZE = 10
MAX_WORKERS = config.MAX_WORKERS_DEFAULT
# =============================================

logger = setup_logger("Step3_LeakageCheck", OUTPUT_DIR)
client = get_openai_client()

def extract_candidates():
    """Step 3.1: Extract all QAs from input files for review."""
    logger.info("Starting extraction of QAs for leakage check...")
    
    if not os.path.exists(INPUT_DIR):
        logger.error(f"Input directory {INPUT_DIR} does not exist.")
        return False

    json_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.json')))
    logger.info(f"Found {len(json_files)} files to process.")

    all_qas_full = []
    llm_qas = []
    global_id = 1
    
    for fpath in json_files:
        filename = os.path.basename(fpath)
        try:
            data = load_json(fpath)
            if not isinstance(data, list):
                continue

            for item in data:
                question = item.get('question', '').strip()
                answer = item.get('answer', '').strip()
                
                # Store full context
                full_entry = {
                    "id": global_id,
                    "file_name": filename,
                    "original_item": item 
                }
                all_qas_full.append(full_entry)
                
                # Store minimal info for LLM
                llm_entry = {
                    "id": global_id,
                    "question": question,
                    "answer": answer
                }
                llm_qas.append(llm_entry)
                global_id += 1
                
        except Exception as e:
            logger.error(f"Error reading {filename}: {e}")

    logger.info(f"Extracted {len(all_qas_full)} QAs.")
    save_json(all_qas_full, FULL_CONTEXT_FILE)
    save_json(llm_qas, LLM_INPUT_FILE)
    return True

def check_batch(batch_data):
    """Process a single batch of QAs with the LLM."""
    batch_id, batch, total_batches = batch_data
    
    prompt_intro = (
        "You are a strict data auditor.\n"
        "Your task is to identify QAs where the Answer is leaked by the Question. "
        "Check if the Answer can be fully inferred or is explicitly stated within the Question text itself, making the QA pair invalid (tautological or leaking).\n"
        "Do NOT consider any external context. Judge strictly based on whether the Question text logically gives away the Answer.\n\n"
        "Here is a batch of QAs. Return a JSON object with a single key 'bad_ids' containing a list of integer IDs for the bad QAs."
    )
    
    qa_text = json.dumps(batch, indent=2, ensure_ascii=False)
        
    prompt = prompt_intro + "\n\nQAs:\n" + qa_text
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that returns JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()
        
        # Parse JSON from content
        try:
            parsed = json.loads(content)
            bad_ids = parsed.get('bad_ids', [])
            if isinstance(bad_ids, list):
                return [int(x) for x in bad_ids if str(x).isdigit() or isinstance(x, int)]
            return []
        except:
            return []
            
    except Exception as e:
        logger.error(f"Error in batch {batch_id}: {e}")
        return []

def run_leakage_check():
    """Step 3.2: Check QAs via API."""
    logger.info("Starting API leakage check...")
    
    if os.path.exists(BAD_IDS_FILE):
        logger.info(f"Found existing bad IDs file at {BAD_IDS_FILE}, skipping check.")
        return True

    qas = load_json(LLM_INPUT_FILE)
    if not qas:
        logger.error("No QAs found for review.")
        return False
        
    total_qas = len(qas)
    batches = [qas[i:i + BATCH_SIZE] for i in range(0, total_qas, BATCH_SIZE)]
    total_batches = len(batches)
    
    logger.info(f"Processing {total_qas} QAs in {total_batches} batches.")
    
    all_bad_ids = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(check_batch, (i, batch, total_batches)): i for i, batch in enumerate(batches)}
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                all_bad_ids.extend(result)
            if i % 10 == 0:
                logger.info(f"Processed {i}/{total_batches} batches...")
                
    save_json(list(set(all_bad_ids)), BAD_IDS_FILE)
    logger.info(f"Check complete. Found {len(set(all_bad_ids))} bad QAs.")
    return True

def remove_bad_qas():
    """Step 3.3: Remove bad QAs and save clean files."""
    logger.info("Starting removal of bad QAs...")
    
    if not os.path.exists(FINAL_CLEAN_DIR):
        os.makedirs(FINAL_CLEAN_DIR)
        
    bad_ids = set()
    if os.path.exists(BAD_IDS_FILE):
        bad_ids = set(load_json(BAD_IDS_FILE))
    
    full_context = load_json(FULL_CONTEXT_FILE)
    if not full_context:
        logger.error("Full context file missing.")
        return False
        
    id_to_item = {item['id']: item for item in full_context}
    
    # Identify what to remove
    file_removals = {}
    qas_to_archive = []
    
    for bib in bad_ids:
        if bib not in id_to_item: continue
        item = id_to_item[bib]
        filename = item.get('file_name')
        original_qa = item.get('original_item')
        q_text = original_qa.get('question', '').strip()
        
        if filename not in file_removals: file_removals[filename] = set()
        file_removals[filename].add(q_text)
        qas_to_archive.append(original_qa)
        
    # Archive to wrong_question.json
    if qas_to_archive:
        current_wrong = {}
        if os.path.exists(WRONG_QA_FILE):
             current_wrong = load_json(WRONG_QA_FILE) or {}
             if isinstance(current_wrong, list): current_wrong = {str(i): x for i,x in enumerate(current_wrong)}

        # Find max id
        start_id = 0
        try:
             keys = [int(k) for k in current_wrong.keys() if str(k).isdigit()]
             if keys: start_id = max(keys)
        except: pass
        
        for qa in qas_to_archive:
            start_id += 1
            current_wrong[str(start_id)] = qa
            
        save_json(current_wrong, WRONG_QA_FILE)
        logger.info(f"Archived {len(qas_to_archive)} QAs to {WRONG_QA_FILE}")

    # Process files
    source_files = glob.glob(os.path.join(INPUT_DIR, '*.json'))
    cleaned_count = 0
    
    for fpath in source_files:
        filename = os.path.basename(fpath)
        out_path = os.path.join(FINAL_CLEAN_DIR, filename)
        
        data = load_json(fpath)
        if not data: continue
        
        new_data = data
        if filename in file_removals:
            bad_set = file_removals[filename]
            new_data = [qa for qa in data if qa.get('question', '').strip() not in bad_set]
            
        save_json(new_data, out_path)
        cleaned_count += 1
        
    logger.info(f"Finished cleaning. Processed {cleaned_count} files.")
    return True

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 1. Extraction
    if not extract_candidates():
        logger.error("Extraction failed.")
        return
        
    # 2. Check
    if not run_leakage_check():
        logger.error("Leakage check failed.")
        return
        
    # 3. Removal
    if not remove_bad_qas():
        logger.error("Removal failed.")
        return
        
    logger.info("Step 3 complete.")

if __name__ == "__main__":
    main()
