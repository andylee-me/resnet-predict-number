#!/usr/bin/env python3
"""
自定義CNN架構從零開始訓練貓狗分類器
每一層都清晰可見，完全自主設計
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import argparse
import os
import time
import copy
import matplotlib.pyplot as plt
import numpy as np

class CustomCNN(nn.Module):
    """
    自定義CNN架構
    清晰的層次結構，每一層都明確定義
    """
    def __init__(self, num_classes=2):
        super(CustomCNN, self).__init__()
        
        print("🏗️ 構建自定義CNN架構...")
        
        # 第一個卷積塊 - 學習基本邊緣和紋理
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 224x224 -> 112x112
        )
        
        # 第二個卷積塊 - 學習更複雜的形狀
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 112x112 -> 56x56
        )
        
        # 第三個卷積塊 - 學習特徵組合
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 56x56 -> 28x28
        )
        
        # 第四個卷積塊 - 學習高級特徵
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        )
        
        # 第五個卷積塊 - 學習抽象特徵
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7
        )
        
        # 全局平均池化 - 替代flatten，更優雅
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 7x7 -> 1x1
        
        # 分類頭 - 最終決策層
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # 初始化權重
        self._initialize_weights()
        
        print("✅ 自定義CNN架構構建完成")
        self._print_model_info()
    
    def _initialize_weights(self):
        """自定義權重初始化"""
        print("🎲 初始化網絡權重...")
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷積層使用He初始化（適合ReLU）
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # BN層初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 全連接層使用Xavier初始化
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"📊 模型參數統計:")
        print(f"   總參數數量: {total_params:,}")
        print(f"   可訓練參數: {trainable_params:,}")
        print(f"   模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    def forward(self, x):
        """前向傳播 - 清晰的數據流"""
        # 輸入: (batch_size, 3, 224, 224)
        
        # 特徵提取階段
        x = self.conv_block1(x)  # (batch_size, 32, 112, 112)
        x = self.conv_block2(x)  # (batch_size, 64, 56, 56)
        x = self.conv_block3(x)  # (batch_size, 128, 28, 28)
        x = self.conv_block4(x)  # (batch_size, 256, 14, 14)
        x = self.conv_block5(x)  # (batch_size, 512, 7, 7)
        
        # 全局平均池化
        x = self.global_avg_pool(x)  # (batch_size, 512, 1, 1)
        x = x.view(x.size(0), -1)    # (batch_size, 512)
        
        # 分類階段
        x = self.classifier(x)       # (batch_size, num_classes)
        
        return x

class CustomCNNTrainer:
    """自定義CNN訓練器"""
    
    def __init__(self, data_dir, target_accuracy=1.0):
        self.data_dir = data_dir
        self.target_accuracy = target_accuracy
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        print(f"🎯 目標訓練準確率: {target_accuracy*100:.1f}%")
        print(f"🔧 使用設備: {self.device}")
        print(f"🏗️ 架構: 自定義CNN (5個卷積塊)")
        
        # 數據預處理
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
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
        self.scheduler = None
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
        
        self.dataloaders = {x: DataLoader(image_datasets[x], batch_size=32,
                                        shuffle=(x == 'train'), num_workers=4)
                          for x in ['train', 'val']}
        
        self.dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        self.class_names = image_datasets['train'].classes
        
        print(f"✅ 訓練集大小: {self.dataset_sizes['train']}")
        print(f"✅ 驗證集大小: {self.dataset_sizes['val']}")
        print(f"✅ 類別: {self.class_names}")
    
    def build_model(self):
        """構建自定義模型"""
        print("🏗️ 構建自定義CNN模型...")
        
        self.model = CustomCNN(num_classes=len(self.class_names))
        self.model = self.model.to(self.device)
        
        # 優化器設置
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # 學習率調度器
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        
        print("✅ 模型構建完成")
    
    def train_to_perfection(self, max_epochs=200):
        """訓練到目標準確率"""
        print(f"🚀 開始訓練自定義CNN到 {self.target_accuracy*100:.1f}% 準確率...")
        print("=" * 60)
        
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        for epoch in range(max_epochs):
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{max_epochs} (LR: {current_lr:.6f})')
            print('-' * 50)
            
            epoch_train_acc = 0.0
            
            # 訓練和驗證階段
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                
                running_loss = 0.0
                running_corrects = 0
                
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
                            # 梯度裁剪
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
                    self.scheduler.step()
                else:
                    val_losses.append(epoch_loss)
                    val_accuracies.append(epoch_acc.cpu().numpy())
                    
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(self.model.state_dict())
            
            # 檢查是否達到目標
            if epoch_train_acc >= self.target_accuracy:
                print(f"\n🎉 達到目標訓練準確率 {self.target_accuracy*100:.1f}%！")
                print(f"實際訓練準確率: {epoch_train_acc*100:.2f}%")
                print(f"在第 {epoch+1} 輪達成目標")
                break
            
            # 進度報告
            if (epoch + 1) % 20 == 0:
                elapsed = time.time() - since
                print(f"📊 進度報告: 訓練準確率 {epoch_train_acc*100:.2f}%, 耗時 {elapsed/60:.1f}分鐘")
            
            print()
        
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')
        
        # 載入最佳模型
        self.model.load_state_dict(best_model_wts)
        
        # 繪製訓練曲線
        self.plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
        
        return self.model
    
    def plot_training_curves(self, train_losses, val_losses, train_accs, val_accs):
        """繪制訓練曲線"""
        plt.figure(figsize=(15, 10))
        
        # Loss曲線
        plt.subplot(2, 2, 1)
        plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
        plt.plot(val_losses, label='Val Loss', color='red', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # 準確率曲線
        plt.subplot(2, 2, 2)
        plt.plot(train_accs, label='Train Acc', color='blue', linewidth=2)
        plt.plot(val_accs, label='Val Acc', color='red', linewidth=2)
        plt.axhline(y=self.target_accuracy, color='green', linestyle='--', 
                   label=f'Target ({self.target_accuracy*100:.0f}%)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        # 放大訓練準確率
        plt.subplot(2, 2, 3)
        plt.plot(train_accs, label='Train Acc', color='blue', linewidth=2, marker='o', markersize=3)
        plt.axhline(y=self.target_accuracy, color='green', linestyle='--',
                   label=f'Target ({self.target_accuracy*100:.0f}%)')
        plt.xlabel('Epoch')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy (Detailed)')
        if max(train_accs) > 0.8:
            plt.ylim(0.7, 1.01)
        plt.legend()
        plt.grid(True)
        
        # 過擬合指標
        plt.subplot(2, 2, 4)
        overfitting = np.array(train_accs) - np.array(val_accs)
        plt.plot(overfitting, color='purple', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Train Acc - Val Acc')
        plt.title('Overfitting Indicator')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('custom_cnn_training_curves.png', dpi=300, bbox_inches='tight')
        print(f"✅ 訓練曲線已保存: custom_cnn_training_curves.png")
        plt.show()
    
    def save_model(self, filepath='best_cat_dog_model.pth'):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'model_architecture': 'CustomCNN',
            'target_accuracy': self.target_accuracy,
            'training_type': 'custom_cnn_from_scratch'
        }, filepath)
        print(f"🎯 自定義CNN模型已保存: {filepath}")

def main():
    parser = argparse.ArgumentParser(description='自定義CNN從零開始訓練')
    parser.add_argument('--data-dir', type=str, default='file/kaggle_cats_vs_dogs_f',
                       help='數據集路徑')
    parser.add_argument('--target-accuracy', type=float, default=1.0,
                       help='目標訓練準確率 (0.0-1.0)')
    parser.add_argument('--max-epochs', type=int, default=200,
                       help='最大訓練輪數')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"❌ 找不到數據路徑: {args.data_dir}")
        return
    
    print("🎯 自定義CNN從零開始訓練器")
    print("=" * 50)
    print(f"📂 數據路徑: {args.data_dir}")
    print(f"🏗️ 模型架構: 自定義CNN")
    print(f"🎯 目標準確率: {args.target_accuracy*100:.1f}%")
    print(f"🔄 最大輪數: {args.max_epochs}")
    
    # 創建訓練器
    trainer = CustomCNNTrainer(args.data_dir, args.target_accuracy)
    
    # 訓練流程
    trainer.load_data()
    trainer.build_model()
    trainer.train_to_perfection(args.max_epochs)
    trainer.save_model('custom_cnn_model.pth')
    
    print("\n🎉 自定義CNN訓練完成！")
    print("\n📋 接下來你可以:")
    print("1. python predict.py --model custom_cnn_model.pth --evaluate-train")
    print("2. 驗證是否達到 100% 訓練準確率")


main()
