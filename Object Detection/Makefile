.PHONY: help install serve test test-fast lint export bench docker docker-up docker-down clean \
        download train eval

PY ?= python

help:
	@echo "PRODUCTION targets (no training):"
	@echo "  make install        install runtime deps"
	@echo "  make serve          run inference server on :8000 (auto-downloads yolo11n.pt)"
	@echo "  make export         export pretrained model to ONNX/TFLite/CoreML for mobile"
	@echo "  make bench          latency benchmark"
	@echo "  make eval           evaluate pretrained on coco128"
	@echo "  make test           pytest (full)"
	@echo "  make test-fast      pytest minus model-loading tests"
	@echo "  make docker-up      start server + Prometheus stack"
	@echo "  make docker-down    stop stack"
	@echo ""
	@echo "OPTIONAL training targets:"
	@echo "  make download       pull a custom Roboflow dataset (needs .env)"
	@echo "  make train          fine-tune YOLO11n on a custom dataset"

# --- Production (no training required) ---

install:
	$(PY) -m pip install -r requirements.txt

serve:
	uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload

export:
	$(PY) export.py --formats onnx tflite coreml --int8 --data coco128.yaml --nms

bench:
	$(PY) benchmark.py --weights yolo11n.pt --image samples/test.jpg

eval:
	$(PY) evaluate.py

test:
	pytest -q

test-fast:
	pytest -q --deselect tests/test_inference.py

lint:
	$(PY) -m pyflakes server tests

docker:
	docker build -t roshdi/object-detection:latest .

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

clean:
	rm -rf runs/ datasets/ results/ __pycache__/ .pytest_cache/

# --- Optional fine-tuning ---

download:
	$(PY) download_dataset.py

train:
	$(PY) train.py --data datasets/COCO-50/data.yaml
