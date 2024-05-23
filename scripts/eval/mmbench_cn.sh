#!/bin/bash

SPLIT="mmbench_dev_cn_20231003"

python -m src.eval.model_vqa_mmbench \
    --model-path playground/model/aligngpt-13b \
    --question-file playground/data/eval/mmbench_cn/$SPLIT.tsv \
    --answers-file playground/data/eval/mmbench_cn/answers/$SPLIT/aligngpt-13b.jsonl \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench_cn/answers_upload_1/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file playground/data/eval/mmbench_cn/$SPLIT.tsv \
    --result-dir playground/data/eval/mmbench_cn/answers/$SPLIT \
    --upload-dir playground/data/eval/mmbench_cn/answers_upload_1/$SPLIT \
    --experiment aligngpt-13b
