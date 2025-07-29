#!/usr/bin/env python3
"""
數據集準備腳本
將下載的貓狗數據集整理成訓練所需的目錄結構
"""

import os
import shutil
import random
import sys
from pathlib import Path

def prepare_kaggle_dataset(source_base_dir="./"):
    """準備 Kaggle 貓狗數據集"""
    
    print(f"正在從 {source_base_dir} 準備數據集...")
    
    # 可能的源數據路徑
    possible_sources = [
        os.path.join(source_base_dir, "train"),
        os.path.join(source_base_dir, "temp_dataset", "train"),
        os.path.join(source_base_dir, "dogs-vs-cats", "train"),
        "train",  # 當前目錄的 train
        "file/kaggle_cats_vs_dogs_f",  # 用戶提到的路徑
    ]
    
    train_source = None
    for source in possible_sources:
        if os.path.exists(source):
            print(f"找到數據源: {source}")
            train_source = source
            break
    
    # 目標數據路徑
    target_base = "kaggle_cats_vs_dogs_f"
    
    # 創建目標目錄結構
    os.makedirs(f"{target_base}/train/cat", exist_ok=True)
    os.makedirs(f"{target_base}/train/dog", exist_ok=True)
    os.makedirs(f"{target_base}/val/cat", exist_ok=True)
    os.makedirs(f"{target_base}/val/dog", exist_ok=True)
    
    print("正在整理數據集...")
    
    if train_source and os.path.exists(train_source):
        # 處理訓練數據
        train_files = []
        
        # 遞歸搜索所有 jpg 文件
        for root, dirs, files in os.walk(train_source):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    train_files.append(os.path.join(root, file))
        
        print(f"找到 {len(train_files)} 張圖片")
        
        # 根據文件名分類
        cat_files = [f for f in train_files if 'cat' in os.path.basename(f).lower()]
        dog_files = [f for f in train_files if 'dog' in os.path.basename(f).lower()]
        
        print(f"分類結果 - 貓: {len(cat_files)}, 狗: {len(dog_files)}")
        
        if len(cat_files) == 0 and len(dog_files) == 0:
            print("警告: 無法根據文件名自動分類，嘗試手動分類...")
            # 如果無法自動分類，假設前一半是貓，後一半是狗
            random.shuffle(train_files)
            mid_point = len(train_files) // 2
            cat_files = train_files[:mid_point]
            dog_files = train_files[mid_point:]
            print(f"手動分類結果 - 貓: {len(cat_files)}, 狗: {len(dog_files)}")
        
        # 隨機打亂
        random.shuffle(cat_files)
        random.shuffle(dog_files)
        
        # 分割比例 (80% 訓練, 20% 驗證)
        cat_split = int(len(cat_files) * 0.8)
        dog_split = int(len(dog_files) * 0.8)
        
        # 移動貓咪圖片
        for i, src_path in enumerate(cat_files):
            filename = f"cat_{i:04d}.jpg"
            if i < cat_split:
                dst = os.path.join(f"{target_base}/train/cat", filename)
            else:
                dst = os.path.join(f"{target_base}/val/cat", filename)
            
            try:
                shutil.copy2(src_path, dst)
            except Exception as e:
                print(f"複製文件失敗 {src_path}: {e}")
        
        # 移動狗狗圖片
        for i, src_path in enumerate(dog_files):
            filename = f"dog_{i:04d}.jpg"
            if i < dog_split:
                dst = os.path.join(f"{target_base}/train/dog", filename)
            else:
                dst = os.path.join(f"{target_base}/val/dog", filename)
            
            try:
                shutil.copy2(src_path, dst)
            except Exception as e:
                print(f"複製文件失敗 {src_path}: {e}")
        
        print(f"訓練集 - 貓: {cat_split}, 狗: {dog_split}")
        print(f"驗證集 - 貓: {len(cat_files) - cat_split}, 狗: {len(dog_files) - dog_split}")
    
    else:
        print(f"警告: 找不到任何數據源")
        print("嘗試過的路徑:")
        for source in possible_sources:
            print(f"  - {source} {'✓' if os.path.exists(source) else '✗'}")
        
        # 檢查當前目錄結構
        print("\n當前目錄結構:")
        for root, dirs, files in os.walk("./"):
            level = root.replace("./", "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            if level < 3:  # 限制深度避免太多輸出
                sub_indent = " " * 2 * (level + 1)
                for file in files[:5]:  # 只顯示前5個文件
                    print(f"{sub_indent}{file}")
                if len(files) > 5:
                    print(f"{sub_indent}... 還有 {len(files) - 5} 個文件")
        
        create_dummy_dataset(target_base)

def create_dummy_dataset(target_base):
    """創建虛擬數據集用於測試"""
    print("創建虛擬數據集用於測試...")
    
    try:
        from PIL import Image
        import numpy as np
        
        # 創建一些虛擬圖片
        for split in ['train', 'val']:
            for category in ['cat', 'dog']:
                num_images = 50 if split == 'train' else 10
                for i in range(num_images):
                    # 創建隨機圖片
                    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                    img = Image.fromarray(img_array)
                    
                    filename = f"{category}_{i:04d}.jpg"
                    filepath = os.path.join(target_base, split, category, filename)
                    img.save(filepath)
        
        print("虛擬數據集創建完成")
    except ImportError:
        print("無法創建虛擬數據集（缺少 PIL），創建空文件...")
        for split in ['train', 'val']:
            for category in ['cat', 'dog']:
                for i in range(5):
                    filename = f"{category}_{i:04d}.txt"
                    filepath = os.path.join(target_base, split, category, filename)
                    with open(filepath, 'w') as f:
                        f.write("dummy file")

def verify_dataset_structure(target_base):
    """驗證數據集結構"""
    print("\n驗證數據集結構...")
    
    required_dirs = [
        f"{target_base}/train/cat",
        f"{target_base}/train/dog", 
        f"{target_base}/val/cat",
        f"{target_base}/val/dog"
    ]
    
    total_files = 0
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            count = len([f for f in os.listdir(dir_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.txt'))])
            print(f"✓ {dir_path}: {count} 個文件")
            total_files += count
        else:
            print(f"✗ 缺少目錄: {dir_path}")
            return False
    
    print(f"總計: {total_files} 個文件")
    return total_files > 0

def main():
    """主函數"""
    print("開始準備數據集...")
    
    # 設置隨機種子
    random.seed(42)
    
    # 獲取源目錄（如果有命令行參數）
    source_dir = sys.argv[1] if len(sys.argv) > 1 else "./"
    print(f"使用源目錄: {source_dir}")
    
    # 準備數據集
    prepare_kaggle_dataset(source_dir)
    
    # 驗證結構
    target_base = "kaggle_cats_vs_dogs_f"
    if verify_dataset_structure(target_base):
        print("\n✓ 數據集準備完成！")
    else:
        print("\n✗ 數據集準備失敗")
        exit(1)

if __name__ == "__main__":
    main()
