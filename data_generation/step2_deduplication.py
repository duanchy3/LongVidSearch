import os
import glob
from utils import setup_logger, save_json, load_json

# ================= Configuration =================
INPUT_DIR = "results/step1_qa_generation"   # Input from Step 1
OUTPUT_DIR = "results/step2_deduplication"  # Output for Step 2
# ===========================================

logger = setup_logger("Step2_Deduplication", OUTPUT_DIR)

def process_single_file(input_file):
    basename = os.path.basename(input_file)
    video_id = basename.replace("_multihop_qa.json", "")
    
    output_file = os.path.join(OUTPUT_DIR, f"{video_id}_deduplicated.json")
    failed_file = os.path.join(OUTPUT_DIR, f"{video_id}_failed_deduplication.json")
    
    logger.info(f"\n{'='*20}\n🧹 Step 2 (Deduplication): {video_id}\n{'='*20}")

    if os.path.exists(output_file):
        logger.info(f"⏩ File exists, skipping.")
        return

    candidates = load_json(input_file)
    if not candidates: return

    final_valid_qas = []
    failed_qas = []
    
    logger.info(f"🚀 Checking {len(candidates)} questions...")

    for idx, qa in enumerate(candidates):
        evidence_slices = qa.get('evidence_slices', [])
        
        # Check for duplicate slice IDs
        if len(evidence_slices) != len(set(evidence_slices)):
            reason = f"Duplicate slice IDs found: {evidence_slices}"
            logger.info(f"Q{idx+1}: ❌ Rejected ({reason})")
            qa['failure_reason'] = reason
            failed_qas.append(qa)
        else:
            final_valid_qas.append(qa)

    save_json(final_valid_qas, output_file)

    if failed_qas:
        save_json(failed_qas, failed_file)
        logger.info(f"📉 Saved {len(failed_qas)} invalid questions to: {failed_file}")

    logger.info(f"🎉 Done. Retention rate: {len(final_valid_qas)}/{len(candidates)}")

if __name__ == "__main__":
    search_pattern = os.path.join(INPUT_DIR, "*_multihop_qa.json")
    target_files = sorted(glob.glob(search_pattern))
    
    logger.info(f"🎯 Step 2 processing {len(target_files)} files")
    
    for f in target_files:
        process_single_file(f)
