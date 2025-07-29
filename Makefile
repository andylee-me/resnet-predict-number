# 貓狗分類器項目 Makefile

.PHONY: help install train predict clean docker-build docker-run setup-data

# 默認目標
help:
	@echo "可用命令:"
	@echo "  install      - 安裝項目依賴"
	@echo "  setup-data   - 準備數據集"
	@echo "  train        - 使用默認參數訓練模型"
	@echo "  train-config - 使用配置文件訓練模型"
	@echo "  predict      - 運行預測示例"
	@echo "  docker-build - 構建 Docker 鏡像"
	@echo "  docker-train - 在 Docker 中訓練"
	@echo "  docker-dev   - 啟動開發環境"
	@echo "  clean        - 清理生成的文件"
	@echo "  test         - 運行測試"

# 安裝依賴
install:
	pip install -r requirements.txt
	@echo "依賴安裝完成！"

# 準備數據集
setup-data:
	python scripts/prepare_dataset.py
	@echo "數據集準備完成！"

# 訓練模型 (默認參數)
train:
	python train_model.py
	@echo "模型訓練完成！"

# 使用配置文件訓練
train-config:
	python train_with_config.py --config config/training_config.yml
	@echo "配置文件訓練完成！"

# 預測示例
predict:
	@if [ ! -f "best_cat_dog_model.pth" ]; then \
		echo "錯誤：找不到訓練好的模型文件！請先運行 make train"; \
		exit 1; \
	fi
	@if [ -d "test_images" ]; then \
		python predict.py --model best_cat_dog_model.pth --folder test_images; \
	else \
		echo "請創建 test_images 資料夾並放入測試圖片"; \
	fi

# 單張圖片預測
predict-single:
	@if [ -z "$(IMAGE)" ]; then \
		echo "用法: make predict-single IMAGE=path/to/image.jpg"; \
		exit 1; \
	fi
	python predict.py --model best_cat_dog_model.pth --image $(IMAGE)

# Docker 操作
docker-build:
	docker build -t cat-dog-classifier .
	@echo "Docker 鏡像構建完成！"

docker-train:
	docker-compose up cat-dog-trainer
	@echo "Docker 訓練完成！"

docker-predict:
	docker-compose up predictor
	@echo "Docker 預測完成！"

docker-dev:
	docker-compose --profile development up notebook
	@echo "開發環境已啟動，請訪問 http://localhost:8888"

# 清理生成的文件
clean:
	rm -f *.pth
	rm -f *.png
	rm -f *.log
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf models/*.pth
	@echo "清理完成！"

# 創建測試數據
create-test-data:
	mkdir -p test_images
	@if [ -d "kaggle_cats_vs_dogs_f/val" ]; then \
		cp kaggle_cats_vs_dogs_f/val/cat/*.jpg test_images/ 2>/dev/null || true; \
		cp kaggle_cats_vs_dogs_f/val/dog/*.jpg test_images/ 2>/dev/null || true; \
		echo "測試數據已創建！"; \
	else \
		echo "請先準備數據集：make setup-data"; \
	fi

# 檢查環境
check-env:
	@echo "檢查 Python 環境..."
	@python --version
	@echo "檢查 PyTorch..."
	@python -c "import torch; print(f'PyTorch 版本: {torch.__version__}')"
	@python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}')"
	@if [ -d "kaggle_cats_vs_dogs_f" ]; then \
		echo "✓ 數據集已準備"; \
	else \
		echo "✗ 數據集未準備，請運行 make setup-data"; \
	fi

# 運行測試
test:
	python -m pytest tests/ -v
	@echo "測試完成！"

# 生成項目報告
report:
	@echo "=== 貓狗分類器項目報告 ==="
	@echo "項目結構:"
	@find . -name "*.py" -o -name "*.yml" -o -name "*.yaml" | head -20
	@echo ""
	@echo "模型文件:"
	@ls -la *.pth 2>/dev/null || echo "無訓練好的模型"
	@echo ""
	@echo "數據集統計:"
	@if [ -d "kaggle_cats_vs_dogs_f" ]; then \
		echo "訓練集貓咪: $$(find kaggle_cats_vs_dogs_f/train/cat -name "*.jpg" 2>/dev/null | wc -l)"; \
		echo "訓練集狗狗: $$(find kaggle_cats_vs_dogs_f/train/dog -name "*.jpg" 2>/dev/null | wc -l)"; \
		echo "驗證集貓咪: $$(find kaggle_cats_vs_dogs_f/val/cat -name "*.jpg" 2>/dev/null | wc -l)"; \
		echo "驗證集狗狗: $$(find kaggle_cats_vs_dogs_f/val/dog -name "*.jpg" 2>/dev/null | wc -l)"; \
	else \
		echo "數據集未準備"; \
	fi

# 快速開始 (全流程)
quickstart: install setup-data train create-test-data predict
	@echo "🎉 快速開始完成！模型已訓練並測試成功！"
