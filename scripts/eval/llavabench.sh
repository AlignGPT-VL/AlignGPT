#!/bin/bash

python -m src.eval.model_vqa \
    --model-path playground/model/aligngpt-13b \
    --question-file playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file playground/data/eval/llava-bench-in-the-wild/answers/aligngpt-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

python src/eval/eval_gpt_review_bench.py \
    --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule src/eval/table/rule.json \
    --answer-list \
        playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/aligngpt-13b.jsonl \
    --output \
        playground/data/eval/llava-bench-in-the-wild/reviews/aligngpt-13b.jsonl

python src/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/aligngpt-13b.jsonl
