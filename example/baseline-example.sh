export OPENAI_API_KEY="<YOUR API KEY>"
python main.py --model gpt-4.1-mini --max_workers 1 \
    --input_ann_file ./full-QA(3000).json \
    --all_cap_file ./video-caption.parquet \
    --output_json ./example/result.json \
    --log_file ./example/result.log