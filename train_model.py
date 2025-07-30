#!/usr/bin/env python3
"""
專門用於達到100%訓練準確率的訓練腳本
通過使用更大模型、更小學習率、更多訓練輪數來實現完全過擬合
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms, models
import argparse
import os
import time
import copy
import matplotlib.pyplot as plt
import numpy as np

class OverfitTrainer:
    def __init__(self, data_dir, target_accuracy=1.0):
        self.data_dir = data_dir
        self.target_accuracy = target_accuracy
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        print(f"🎯 目標訓練準確率: {target_accuracy*100:.1f}%")
        print(f"🔧 使用設備: {self.device}")
        
        # 針對過擬合的數據變換（減少隨機性）
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),  # 使用中心裁剪而非隨機裁剪
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.dataloaders = {}
        self.dataset_sizes = {}
        self.class_names = []
        
    def load_data(self):
        """加載數據"""
        print("📂 正在加載數據...")
        
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x),
                                                self.data_transforms[x])
                         for x in ['train', 'val']}
        
        # 使用較小的batch size以獲得更精確的梯度
        self.dataloaders = {x: DataLoader(image_datasets[x], batch_size=8,
                                        shuffle=(x == 'train'), num_workers=4)
                          for x in ['train', 'val']}
        
        self.dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        self.class_names = image_datasets['train'].classes
        
        print(f"✅ 訓練集大小: {self.dataset_sizes['train']}")
        print(f"✅ 驗證集大小: {self.dataset_sizes['val']}")
        print(f"✅ 類別: {self.class_names}")
        
    def build_model(self, architecture='resnet50'):
        """構建更大容量的模型"""
        print(f"🏗️ 正在構建模型: {architecture}")
        
        if architecture == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, 2)
        elif architecture == 'resnet101':
            self.model = models.resnet101(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, 2)
        elif architecture == 'resnet34':
            self.model = models.resnet34(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, 2)
        else:  # resnet18
            self.model = models.resnet18(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, 2)
        
        # 解凍所有層進行訓練（不凍結任何層）
        for param in self.model.parameters():
            param.requires_grad = True
        
        self.model = self.model.to(self.device)
        
        # 使用非常小的學習率以實現精確擬合
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00001, weight_decay=0.0)
        
        print(f"✅ 模型已構建，所有層均可訓練")
        
    def train_to_perfection(self, max_epochs=200):
        """訓練直到達到目標準確率"""
        print(f"🚀 開始訓練到 {self.target_accuracy*100:.1f}% 準確率...")
        print(f"🔄 最大訓練輪數: {max_epochs}")
        print("=" * 60)
        
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        epochs_without_improvement = 0
        max_patience = 30
        
        for epoch in range(max_epochs):
            print(f'Epoch {epoch+1}/{max_epochs}')
            print('-' * 40)
            
            # 每個epoch都有訓練和驗證階段
            epoch_train_acc = 0.0
            epoch_val_acc = 0.0
            
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                
                running_loss = 0.0
                running_corrects = 0
                
                # 遍歷數據
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    self.optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                
                print(f'{phase} Loss: {epoch_loss:.6f} Acc: {epoch_acc:.4f} ({epoch_acc*100:.2f}%)')
                
                if phase == 'train':
                    train_losses.append(epoch_loss)
                    train_accuracies.append(epoch_acc.cpu().numpy())
                    epoch_train_acc = epoch_acc
                else:
                    val_losses.append(epoch_loss)
                    val_accuracies.append(epoch_acc.cpu().numpy())
                    epoch_val_acc = epoch_acc
                
                # 保存最佳模型
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    epochs_without_improvement = 0
                elif phase == 'val':
                    epochs_without_improvement += 1
            
            # 檢查是否達到目標訓練準確率
            if epoch_train_acc >= self.target_accuracy:
                print(f"\n🎉 達到目標訓練準確率 {self.target_accuracy*100:.1f}%！")
                print(f"實際訓練準確率: {epoch_train_acc*100:.2f}%")
                print(f"在第 {epoch+1} 輪達成目標")
                break
            
            # 早停機制（但主要關注訓練準確率）
            if epochs_without_improvement >= max_patience:
                print(f"\n⏰ 驗證準確率 {max_patience} 輪無改善，但繼續追求訓練準確率...")
                # 不停止訓練，繼續追求100%訓練準確率
            
            print()
        
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')
        
        # 如果沒有達到目標，使用當前模型
        if epoch_train_acc < self.target_accuracy:
            print(f"⚠️ 未完全達到目標，最終訓練準確率: {epoch_train_acc*100:.2f}%")
            self.model.load_state_dict(self.model.state_dict())  # 使用最後的模型
        else:
            self.model.load_state_dict(self.model.state_dict())  # 使用達成目標的模型
        
        # 繪製訓練曲線
        self.plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
        
        return self.model
    
    def plot_training_curves(self, train_losses, val_losses, train_accs, val_accs):
        """繪製訓練曲線"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(val_losses, label='Val Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(train_accs, label='Train Acc', color='blue')
        plt.plot(val_accs, label='Val Acc', color='red')
        plt.axhline(y=self.target_accuracy, color='green', linestyle='--', label=f'Target ({self.target_accuracy*100:.0f}%)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        # 放大訓練準確率曲線
        plt.plot(train_accs, label='Train Acc', color='blue', linewidth=2)
        plt.axhline(y=self.target_accuracy, color='green', linestyle='--', label=f'Target ({self.target_accuracy*100:.0f}%)')
        plt.xlabel('Epoch')
        plt.ylabel('Training Accuracy')
        plt.ylim(0.9, 1.01)  # 放大到90%-100%區間
        plt.legend()
        plt.title('Training Accuracy (Zoomed)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('overfit_training_curves.png', dpi=300, bbox_inches='tight')
        print(f"✅ 訓練曲線已保存到: overfit_training_curves.png")
        plt.show()
    
    def save_model(self, filepath='perfect_cat_dog_model.pth'):
        """保存達到100%準確率的模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'model_architecture': 'resnet50_overfitted',
            'target_accuracy': self.target_accuracy,
            'training_type': 'overfitted_for_perfect_accuracy'
        }, filepath)
        print(f"🎯 完美擬合模型已保存到: {filepath}")

def main():
    parser = argparse.ArgumentParser(description='訓練100%準確率的貓狗分類器')
    parser.add_argument('--data-dir', type=str, default='kaggle_cats_vs_dogs_f',
                       help='數據集路徑')
    parser.add_argument('--architecture', type=str, default='resnet50',
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'],
                       help='模型架構')
    parser.add_argument('--target-accuracy', type=float, default=1.0,
                       help='目標訓練準確率 (0.0-1.0)')
    parser.add_argument('--max-epochs', type=int, default=200,
                       help='最大訓練輪數')
    
    args = parser.parse_args()
    
    # 檢查數據路徑
    if not os.path.exists(args.data_dir):
        print(f"❌ 找不到數據路徑: {args.data_dir}")
        return
    
    print("🎯 100% 訓練準確率專用訓練器")
    print("=" * 50)
    print(f"📂 數據路徑: {args.data_dir}")
    print(f"🏗️ 模型架構: {args.architecture}")
    print(f"🎯 目標準確率: {args.target_accuracy*100:.1f}%")
    print(f"🔄 最大輪數: {args.max_epochs}")
    
    # 創建訓練器
    trainer = OverfitTrainer(args.data_dir, args.target_accuracy)
    
    # 訓練流程
    trainer.load_data()
    trainer.build_model(args.architecture)
    trainer.train_to_perfection(args.max_epochs)
    trainer.save_model('perfect_cat_dog_model.pth')
    
    print("\n🎉 訓練完成！")
    print("\n📋 接下來你可以:")
    print("1. 使用 python predict.py --model perfect_cat_dog_model.pth --evaluate-train")
    print("2. 驗證是否達到 100% 訓練準確率")

if __name__ == '__main__':
    main()
