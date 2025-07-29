#!/usr/bin/env python3
"""
數據集準備腳本
將下載的貓狗數據集整理成訓練所需的目錄結構
"""

import os
import shutil
import random
from pathlib import Path

def prepare_kaggle_dataset():
    """準備 Kaggle 貓狗數據集"""
    
    # 源數據路徑
    train_source = "train"  # Kaggle 解壓後的訓練數據
    test_source = "test1"   # Kaggle 解壓後的測試數據
    
    # 目標數據路徑
    target_base = "kaggle_cats_vs_dogs_f"
    
    # 創建目標目錄結構
    os.makedirs(f"{target_base}/train/cat", exist_ok=True)
    os.makedirs(f"{target_base}/train/dog", exist_ok=True)
    os.makedirs(f"{target_base}/val/cat", exist_ok=True)
    os.makedirs(f"{target_base}/val/dog", exist_ok=True)
    
    print("正在整理數據集...")
    
    if os.path.exists(train_source):
        # 處理訓練數據
        train_files = os.listdir(train_source)
        cat_files = [f for f in train_files if f.startswith('cat.')]
        dog_files = [f for f in train_files if f.startswith('dog.')]
        
        print(f"找到 {len(cat_files)} 張貓咪圖片")
        print(f"找到 {len(dog_files)} 張狗狗圖片")
        
        # 隨機打亂
        random.shuffle(cat_files)
        random.shuffle(dog_files)
        
        # 分割比例 (80% 訓練, 20% 驗證)
        cat_split = int(len(cat_files) * 0.8)
        dog_split = int(len(dog_files) * 0.8)
        
        # 移動貓咪圖片
        for i, filename in enumerate(cat_files):
            src = os.path.join(train_source, filename)
            if i < cat_split:
                dst = os.path.join(f"{target_base}/train/cat", filename)
            else:
                dst = os.path.join(f"{target_base}/val/cat", filename)
            shutil.copy2(src, dst)
        
        # 移動狗狗圖片
        for i, filename in enumerate(dog_files):
            src = os.path.join(train_source, filename)
            if i < dog_split:
                dst = os.path.join(f"{target_base}/train/dog", filename)
            else:
                dst = os.path.join(f"{target_base}/val/dog", filename)
            shutil.copy2(src, dst)
        
        print(f"訓練集 - 貓: {cat_split}, 狗: {dog_split}")
        print(f"驗證集 - 貓: {len(cat_files) - cat_split}, 狗: {len(dog_files) - dog_split}")
    
    else:
        print(f"警告: 找不到源數據目錄 {train_source}")
        
        # 嘗試其他可能的數據源
        possible_sources = [
            "dogs-vs-cats/train",
            "dogs-vs-cats-redux-kernels-edition/train", 
            "cat-and-dog/training_set",
            "PetImages"
        ]
        
        for source in possible_sources:
            if os.path.exists(source):
                print(f"找到替代數據源: {source}")
                prepare_alternative_dataset(source, target_base)
                break
        else:
            print("未找到任何可用的數據源")
            create_dummy_dataset(target_base)

def prepare_alternative_dataset(source_dir, target_base):
    """處理其他格式的數據集"""
    print(f"正在處理數據源: {source_dir}")
    
    # 遞歸搜索所有圖片文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    all_files = []
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                full_path = os.path.join(root, file)
                all_files.append((full_path, file))
    
    # 根據文件名分類
    cat_files = [(path, name) for path, name in all_files 
                 if 'cat' in name.lower()]
    dog_files = [(path, name) for path, name in all_files 
                 if 'dog' in name.lower()]
    
    print(f"分類結果 - 貓: {len(cat_files)}, 狗: {len(dog_files)}")
    
    # 複製文件
    copy_files_with_split(cat_files, target_base, 'cat')
    copy_files_with_split(dog_files, target_base, 'dog')

def copy_files_with_split(file_list, target_base, category):
    """複製文件並分割為訓練集和驗證集"""
    if not file_list:
        print(f"警告: 沒有找到 {category} 的圖片")
        return
    
    random.shuffle(file_list)
    split_idx = int(len(file_list) * 0.8)
    
    # 訓練集
    for i, (src_path, filename) in enumerate(file_list[:split_idx]):
        dst_path = os.path.join(target_base, 'train', category, f"{category}_{i:04d}.jpg")
        shutil.copy2(src_path, dst_path)
    
    # 驗證集
    for i, (src_path, filename) in enumerate(file_list[split_idx:]):
        dst_path = os.path.join(target_base, 'val', category, f"{category}_val_{i:04d}.jpg")
        shutil.copy2(src_path, dst_path)
    
    print(f"{category}: 訓練集 {split_idx}, 驗證集 {len(file_list) - split_idx}")

def create_dummy_dataset(target_base):
    """創建虛擬數據集用於測試"""
    print("創建虛擬數據集用於測試...")
    
    from PIL import Image
    import numpy as np
    
    # 創建一些虛擬圖片
    for split in ['train', 'val']:
        for category in ['cat', 'dog']:
            num_images = 10 if split == 'train' else 3
            for i in range(num_images):
                # 創建隨機圖片
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                
                filename = f"{category}_{i:04d}.jpg"
                filepath = os.path.join(target_base, split, category, filename)
                img.save(filepath)
    
    print("虛擬數據集創建完成")

def verify_dataset_structure(target_base):
    """驗證數據集結構"""
    print("\n驗證數據集結構...")
    
    required_dirs = [
        f"{target_base}/train/cat",
        f"{target_base}/train/dog", 
        f"{target_base}/val/cat",
        f"{target_base}/val/dog"
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            count = len([f for f in os.listdir(dir_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"✓ {dir_path}: {count} 張圖片")
        else:
            print(f"✗ 缺少目錄: {dir_path}")
            return False
    
    return True

def main():
    """主函數"""
    print("開始準備數據集...")
    
    # 設置隨機種子
    random.seed(42)
    
    # 準備數據集
    prepare_kaggle_dataset()
    
    # 驗證結構
    target_base = "kaggle_cats_vs_dogs_f"
    if verify_dataset_structure(target_base):
        print("\n✓ 數據集準備完成！")
    else:
        print("\n✗ 數據集準備失敗")
        exit(1)

if __name__ == "__main__":
    main()
