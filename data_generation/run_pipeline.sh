#!/bin/bash
set -e

# Data Construction Pipeline
# This script runs the sequence of data generation and cleaning steps.

# Step 1: QA Generation (Multihop QA)
echo "========================================="
echo "Running Step 1: QA Generation"
echo "========================================="
python3 step1_qa_generation.py

# Step 2: Deduplication
echo "========================================="
echo "Running Step 2: Deduplication"
echo "========================================="
python3 step2_deduplication.py

# Step 3: Leakage Check (New step inserted)
# Extracts candidates, checks with API, and removes leaking QAs
echo "========================================="
echo "Running Step 3: Leakage Check & Cleaning"
echo "========================================="
python3 step3_leakage_check.py

# Step 4: Logic Check
echo "========================================="
echo "Running Step 4: Logic Check"
echo "========================================="
python3 step4_logic_check.py

# Step 5: Necessity Check
echo "========================================="
echo "Running Step 5: Necessity Check"
echo "========================================="
python3 step5_necessity_check.py

# Step 6: Video Verification
echo "========================================="
echo "Running Step 6: Video Verification"
echo "========================================="
python3 step6_video_verification.py

echo "Pipeline completed successfully!"
