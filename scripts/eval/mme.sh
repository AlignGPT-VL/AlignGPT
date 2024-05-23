#!/bin/bash

python -m src.eval.model_vqa_loader \
    --model-path playground/model/aligngpt-13b \
    --question-file playground/data/eval/MME/llava_mme_1.jsonl \
    --image-folder playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file playground/data/eval/MME/answers/aligngpt-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd playground/data/eval/MME

python convert_answer_to_mme.py --experiment aligngpt-13b

cd eval_tool

python calculation.py --results_dir playground/data/eval/MME/MME_Benchmark_release_version/eval_tool/answers/aligngpt-13b
