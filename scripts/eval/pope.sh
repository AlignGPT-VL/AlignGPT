#!/bin/bash

python -m src.eval.model_vqa_loader \
    --model-path playground/model/aligngpt-13b \
    --question-file playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder playground/data/eval/pope/val2014 \
    --answers-file playground/data/eval/pope/answers/aligngpt-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python src/eval/eval_pope.py \
    --annotation-dir playground/data/eval/pope/coco \
    --question-file playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file playground/data/eval/pope/answers/aligngpt-13b.jsonl
