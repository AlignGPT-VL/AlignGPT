#!/bin/bash

python -m src.eval.model_vqa_science \
    --model-path playground/model/aligngpt-13b \
    --question-file playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder playground/data/eval/scienceqa/ScienceQA_DATA/test \
    --answers-file playground/data/eval/scienceqa/answers/aligngpt-13b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python src/eval/eval_science_qa.py \
    --base-dir playground/data/eval/scienceqa/ScienceQA_DATA \
    --result-file playground/data/eval/scienceqa/answers/aligngpt-13b.jsonl \
    --output-file playground/data/eval/scienceqa/answers/aligngpt-13b_output.jsonl \
    --output-result playground/data/eval/scienceqa/answers/aligngpt-13b_result.json
