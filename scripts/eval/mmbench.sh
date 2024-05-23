#!/bin/bash

SPLIT="mmbench_dev_20230712"

python -m src.eval.model_vqa_mmbench \
    --model-path playground/model/aligngpt-13b \
    --question-file playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file playground/data/eval/mmbench/answers/$SPLIT/aligngpt-13b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir playground/data/eval/mmbench/answers_upload_1/$SPLIT \
    --experiment aligngpt-13b
