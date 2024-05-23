#!/bin/bash

python -m src.eval.model_vqa_loader \
    --model-path playground/model/aligngpt-13b \
    --question-file playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder playground/data/eval/vizwiz/test \
    --answers-file playground/data/eval/vizwiz/answers/aligngpt-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file playground/data/eval/vizwiz/answers/aligngpt-13b.jsonl \
    --result-upload-file playground/data/eval/vizwiz/answers_upload_1/aligngpt-13b.json
