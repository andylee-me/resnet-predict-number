#!/usr/bin/env python3
"""
手寫ResNet架構實現
完全透明的ResNet邏輯，每一個殘差塊都清晰可見
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

class BasicBlock(nn.Module):
    """
    ResNet基礎殘差塊
    包含兩個3x3卷積層和一個跳躍連接
    """
    expansion = 1  # 通道數擴展係數
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        # 第一個卷積層
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 第二個卷積層
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 跳躍連接的下採樣層（如果需要）
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        # 保存輸入用於跳躍連接
        identity = x
        
        # 第一個卷積-BN-ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # 第二個卷積-BN
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 跳躍連接：如果維度不匹配需要下採樣
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # 殘差連接：F(x) + x
        out += identity
        out = F.relu(out)
        
        return out

class HandwrittenResNet(nn.Module):
    """
    手寫的ResNet架構
    清晰展示每一層的構造邏輯
    """
    
    def __init__(self, block, layers, num_classes=2, zero_init_residual=False):
        super(HandwrittenResNet, self).__init__()
        
        print("🏗️ 構建手寫ResNet架構...")
        
        self.in_channels = 64
        
        # 第一層：7x7卷積 + BatchNorm + ReLU + MaxPool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 四個殘差層組
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 全局平均池化和分類器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 權重初始化
        self._initialize_weights(zero_init_residual)
        
        print("✅ 手寫ResNet架構構建完成")
        self._print_architecture_info(block, layers)
    
    def _make_layer(self, block, channels, blocks, stride=1):
        """
        構建一個殘差層組
        block: BasicBlock類
        channels: 輸出通道數
        blocks: 該層組中的block數量
        stride: 第一個block的步長
        """
        downsample = None
        
        # 如果步長不為1或者輸入輸出通道數不匹配，需要下採樣
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion),
            )
        
        layers = []
        # 第一個block可能需要下採樣
        layers.append(block(self.in_channels, channels, stride, downsample))
        self.in_channels = channels * block.expansion
        
        # 後續block保持相同維度
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self, zero_init_residual):
        """權重初始化"""
        print("🎲 初始化ResNet權重...")
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def _print_architecture_info(self, block, layers):
        """打印架構信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"📊 ResNet架構信息:")
        print(f"   層配置: {layers}")
        print(f"   總參數數量: {total_params:,}")
        print(f"   可訓練參數: {trainable_params:,}")
        print(f"   模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        # 計算每層的輸出尺寸
        print(f"📐 各層輸出尺寸:")
        print(f"   輸入圖片: (3, 224, 224)")
        print(f"   conv1 + pool: (64, 56, 56)")
        print(f"   layer1: ({64 * block.expansion}, 56, 56)")
        print(f"   layer2: ({128 * block.expansion}, 28, 28)")
        print(f"   layer3: ({256 * block.expansion}, 14, 14)")
        print(f"   layer4: ({512 * block.expansion}, 7, 7)")
        print(f"   avgpool: ({512 * block.expansion}, 1, 1)")
        print(f"   fc: (2,)")
    
    def forward(self, x):
        """前向傳播 - 展示完整的數據流"""
        # 輸入: (batch_size, 3, 224, 224)
        
        # 初始卷積層
        x = self.conv1(x)    # (batch_size, 64, 112, 112)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # (batch_size, 64, 56, 56)
        
        # 四個殘差層組
        x = self.layer1(x)   # (batch_size, 64, 56, 56)
        x = self.layer2(x)   # (batch_size, 128, 28, 28)
        x = self.layer3(x)   # (batch_size, 256, 14, 14)
        x = self.layer4(x)   # (batch_size, 512, 7, 7)
        
        # 全局平均池化
        x = self.avgpool(x)  # (batch_size, 512, 1, 1)
        x = x.view(x.size(0), -1)  # (batch_size, 512)
        
        # 分類器
        x = self.fc(x)       # (batch_size, 2)
        
        return x

def resnet18(num_classes=2, **kwargs):
    """構建ResNet-18"""
    return HandwrittenResNet(BasicBlock, [2, 2, 2, 2], num_classes, **kwargs)

def resnet34(num_classes=2, **kwargs):
    """構建ResNet-34"""
    return HandwrittenResNet(BasicBlock, [3, 4, 6, 3], num_classes, **kwargs)

class HandwrittenResNetTrainer:
    """手寫ResNet訓練器"""
    
    def __init__(self, data_dir, architecture='resnet34', target_accuracy=1.0):
        self.data_dir = data_dir
        self.architecture = architecture
        self.target_accuracy = target_accuracy
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        print(f"🎯 目標訓練準確率: {target_accuracy*100:.1f}%")
        print(f"🔧 使用設備: {self.device}")
        print(f"🏗️ 架構: 手寫{architecture.upper()}")
        
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
        """構建手寫ResNet模型"""
        print(f"🏗️ 構建手寫{self.architecture.upper()}...")
        
        if self.architecture == 'resnet18':
            self.model = resnet18(num_classes=len(self.class_names))
        elif self.architecture == 'resnet34':
            self.model = resnet34(num_classes=len(self.class_names))
        else:
            raise ValueError(f"不支持的架構: {self.architecture}")
        
        self.model = self.model.to(self.device)
        
        # 優化器設置
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # 學習率調度器
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=60, gamma=0.1)
        
        print("✅ 手寫ResNet模型構建完成")
    
    def train_to_perfection(self, max_epochs=250):
        """訓練到目標準確率"""
        print(f"🚀 開始訓練手寫ResNet到 {self.target_accuracy*100:.1f}% 準確率...")
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
            if (epoch + 1) % 25 == 0:
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
        plt.figure(figsize=(16, 12))
        
        # Loss曲線
        plt.subplot(2, 3, 1)
        plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
        plt.plot(val_losses, label='Val Loss', color='red', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        
        # 準確率曲線
        plt.subplot(2, 3, 2)
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
        plt.subplot(2, 3, 3)
        plt.plot(train_accs, label='Train Acc', color='blue', linewidth=2, marker='o', markersize=2)
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
        plt.subplot(2, 3, 4)
        overfitting = np.array(train_accs) - np.array(val_accs)
        plt.plot(overfitting, color='purple', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Train Acc - Val Acc')
        plt.title('Overfitting Indicator')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True)
        
        # 最近50輪的準確率
        plt.subplot(2, 3, 5)
        recent_epochs = min(50, len(train_accs))
        recent_accs = train_accs[-recent_epochs:]
        plt.plot(range(len(train_accs)-recent_epochs, len(train_accs)), recent_accs, 
                color='blue', linewidth=2, marker='o', markersize=3)
        plt.axhline(y=self.target_accuracy, color='green', linestyle='--',
                   label=f'Target ({self.target_accuracy*100:.0f}%)')
        plt.xlabel('Epoch')
        plt.ylabel('Training Accuracy')
        plt.title(f'Recent {recent_epochs} Epochs')
        plt.legend()
        plt.grid(True)
        
        # 模型架構信息
        plt.subplot(2, 3, 6)
        plt.text(0.1, 0.8, f'Architecture: {self.architecture.upper()}', fontsize=12, fontweight='bold')
        plt.text(0.1, 0.7, f'Target Accuracy: {self.target_accuracy*100:.1f}%', fontsize=11)
        plt.text(0.1, 0.6, f'Final Train Acc: {train_accs[-1]*100:.2f}%', fontsize=11)
        plt.text(0.1, 0.5, f'Final Val Acc: {val_accs[-1]*100:.2f}%', fontsize=11)
        plt.text(0.1, 0.4, f'Total Epochs: {len(train_accs)}', fontsize=11)
        plt.text(0.1, 0.3, f'Best Val Acc: {max(val_accs)*100:.2f}%', fontsize=11)
        plt.text(0.1, 0.2, f'Training Mode: From Scratch', fontsize=11, style='italic')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('Training Summary')
        
        plt.tight_layout()
        plt.savefig('handwritten_resnet_training_curves.png', dpi=300, bbox_inches='tight')
        print(f"✅ 訓練曲線已保存: handwritten_resnet_training_curves.png")
        plt.show()
    
    def save_model(self, filepath='best_cat_dog_model.pth'):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'model_architecture': f'Handwritten{self.architecture.upper()}',
            'target_accuracy': self.target_accuracy,
            'training_type': 'best_cat_dog_model'
        }, filepath)
        print(f"🎯 手寫ResNet模型已保存: {filepath}")

def main():
    parser = argparse.ArgumentParser(description='手寫ResNet從零開始訓練')
    parser.add_argument('--data-dir', type=str, default='file/kaggle_cats_vs_dogs_f',
                       help='數據集路徑')
    parser.add_argument('--architecture', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34'],
                       help='ResNet架構')
    parser.add_argument('--target-accuracy', type=float, default=1.0,
                       help='目標訓練準確率 (0.0-1.0)')
    parser.add_argument('--max-epochs', type=int, default=250,
                       help='最大訓練輪數')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"❌ 找不到數據路徑: {args.data_dir}")
        return
    
    print("🎯 手寫ResNet從零開始訓練器")
    print("=" * 50)
    print(f"📂 數據路徑: {args.data_dir}")
    print(f"🏗️ 模型架構: 手寫{args.architecture.upper()}")
    print(f"🎯 目標準確率: {args.target_accuracy*100:.1f}%")
    print(f"🔄 最大輪數: {args.max_epochs}")
    
    # 創建訓練器
    trainer = HandwrittenResNetTrainer(args.data_dir, args.architecture, args.target_accuracy)
    
    # 訓練流程
    trainer.load_data()
    trainer.build_model()
    trainer.train_to_perfection(args.max_epochs)
    trainer.save_model('best_cat_dog_model.pth')
    
    print("\n🎉 手寫ResNet訓練完成！")
    print("\n📋 接下來你可以:")
    print("1. python predict.py --model best_cat_dog_model.pth --evaluate-train")
    print("2. 驗證是否達到 100% 訓練準確率")


main()
