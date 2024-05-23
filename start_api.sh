export http_proxy=
export https_proxy=
export all_proxy=
nohup python -m src.serve.controller --host 0.0.0.0 --port 10000 > /dev/null 2>&1 &
nohup python -m src.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload > /dev/null 2>&1 &
nohup python -m src.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path playground/model/aligngpt-13b > /dev/null 2>&1 &