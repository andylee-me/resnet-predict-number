FROM python:3.9-slim

# 設置工作目錄
WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 複製需求文件
COPY requirements.txt .

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 複製項目文件
COPY . .

# 創建數據目錄
RUN mkdir -p kaggle_cats_vs_dogs_f/train/cat \
    kaggle_cats_vs_dogs_f/train/dog \
    kaggle_cats_vs_dogs_f/val/cat \
    kaggle_cats_vs_dogs_f/val/dog

# 設置權限
RUN chmod +x scripts/prepare_dataset.py

# 暴露端口（如果需要 web 服務）
EXPOSE 8000

# 默認命令
CMD ["python", "train_model.py"]
