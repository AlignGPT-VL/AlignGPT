# Evaluation

### VQAv2

- Multi-GPU inference:
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/vqav2.sh
```
- Submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/830/submission): `./playground/data/eval/vqav2/answers_upload`.

### GQA

- Multi-GPU inference:
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/gqa.sh
```

### VisWiz

- Single-GPU inference:
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/vizwiz.sh
```
- Submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/2185/submission): `./playground/data/eval/vizwiz/answers_upload`.

### ScienceQA-IMG

- Single-GPU inference and evaluate:
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/sqa.sh
```

### TextVQA

- Single-GPU inference and evaluate:
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/textvqa.sh
```

### POPE

- Single-GPU inference and evaluate:
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/pope.sh
```

### MME

- Single-GPU inference and evaluate:
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mme.sh
```

### MMBench

- Single-GPU inference:
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mmbench.sh
```
- Submit the results to the [evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission): `./playground/data/eval/mmbench/answers_upload/mmbench_dev_20230712`.

### MMBench-CN

- Single-GPU inference:
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mmbench_cn.sh
```
- Submit the results to the [evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission): `./playground/data/eval/mmbench/answers_upload/mmbench_dev_cn_20231003`.


### SEED-Bench (Image)

- Multiple-GPU inference and evaluate:
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/seed.sh
```

### LLaVA-Bench-in-the-Wild

- Single-GPU inference and evaluate:
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/llavabench.sh
```

### MM-Vet

- Single-GPU inference:
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mmvet.sh
```
- Evaluate the predictions in `./playground/data/eval/mmvet/results` using the official jupyter notebook.
