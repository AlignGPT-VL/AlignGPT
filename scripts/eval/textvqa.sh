#!/bin/bash

python -m src.eval.model_vqa_loader \
    --model-path playground/model/aligngpt-13b \
    --question-file playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder playground/data/eval/textvqa/train_images \
    --answers-file playground/data/eval/textvqa/answers/aligngpt-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m src.eval.eval_textvqa \
    --annotation-file playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file playground/data/eval/textvqa/answers/aligngpt-13b.jsonl
