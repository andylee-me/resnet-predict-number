import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms, models
import yaml
import argparse
import os
import time
import copy
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class ConfigurableTrainer:
    def __init__(self, config_path):
        """使用配置文件初始化訓練器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = self._setup_device()
        self.data_transforms = self._setup_transforms()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        self.dataloaders = {}
        self.dataset_sizes = {}
        self.class_names = []
        self.best_acc = 0.0
        self.early_stopping_counter = 0
        
    def _setup_device(self):
        """設置計算設備"""
        device_config = self.config['hardware']['device']
        if device_config == 'auto':
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_config)
        
        print(f"使用設備: {device}")
        return device
    
    def _setup_transforms(self):
        """根據配置設置數據變換"""
        aug_config = self.config['augmentation']
        
        # 訓練時的變換
        train_transforms = []
        if aug_config['train']['random_resized_crop']:
            train_transforms.append(transforms.RandomResizedCrop(self.config['data']['image_size']))
        if aug_config['train']['random_horizontal_flip']:
            train_transforms.append(transforms.RandomHorizontalFlip())
        if aug_config['train']['random_rotation']:
            train_transforms.append(transforms.RandomRotation(aug_config['train']['random_rotation']))
        
        # 顏色抖動
        if 'color_jitter' in aug_config['train']:
            cj = aug_config['train']['color_jitter']
            train_transforms.append(transforms.ColorJitter(
                brightness=cj['brightness'],
                contrast=cj['contrast'],
                saturation=cj['saturation'],
                hue=cj['hue']
            ))
        
        train_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                aug_config['train']['normalize']['mean'],
                aug_config['train']['normalize']['std']
            )
        ])
        
        # 驗證時的變換
        val_transforms = [
            transforms.Resize(aug_config['val']['resize']),
            transforms.CenterCrop(aug_config['val']['center_crop']),
            transforms.ToTensor(),
            transforms.Normalize(
                aug_config['val']['normalize']['mean'],
                aug_config['val']['normalize']['std']
            )
        ]
        
        return {
            'train': transforms.Compose(train_transforms),
            'val': transforms.Compose(val_transforms)
        }
    
    def load_data(self):
        """加載數據"""
        data_config = self.config['data']
        print("正在加載數據...")
        
        image_datasets = {x: datasets.ImageFolder(
            os.path.join(data_config['data_dir'], x),
            self.data_transforms[x]
        ) for x in ['train', 'val']}
        
        self.dataloaders = {x: DataLoader(
            image_datasets[x],
            batch_size=data_config['batch_size'],
            shuffle=True,
            num_workers=data_config['num_workers']
        ) for x in ['train', 'val']}
        
        self.dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        self.class_names = image_datasets['train'].classes
        
        print(f"訓練集大小: {self.dataset_sizes['train']}")
        print(f"驗證集大小: {self.dataset_sizes['val']}")
        print(f"類別: {self.class_names}")
    
    def build_model(self):
        """根據配置構建模型"""
        model_config = self.config['model']
        print(f"正在構建模型: {model_config['architecture']}")
        
        # 根據架構選擇模型
        if model_config['architecture'] == 'resnet18':
            self.model = models.resnet18(pretrained=model_config['pretrained'])
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, model_config['num_classes'])
        elif model_config['architecture'] == 'resnet34':
            self.model = models.resnet34(pretrained=model_config['pretrained'])
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, model_config['num_classes'])
        elif model_config['architecture'] == 'resnet50':
            self.model = models.resnet50(pretrained=model_config['pretrained'])
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, model_config['num_classes'])
        else:
            raise ValueError(f"不支持的模型架構: {model_config['architecture']}")
        
        # 是否凍結骨幹網絡
        if model_config['freeze_backbone']:
            for param in self.model.parameters():
                param.requires_grad = False
            # 只讓最後一層可訓練
            for param in self.model.fc.parameters():
                param.requires_grad = True
        
        self.model = self.model.to(self.device)
        
        # 設置優化器
        self._setup_optimizer()
        
    def _setup_optimizer(self):
        """設置優化器和學習率調度器"""
        train_config = self.config['training']
        
        # 選擇需要優化的參數
        if self.config['model']['freeze_backbone']:
            params_to_update = self.model.fc.parameters()
        else:
            params_to_update = self.model.parameters()
        
        # 選擇優化器
        if train_config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                params_to_update,
                lr=train_config['learning_rate'],
                weight_decay=train_config['weight_decay']
            )
        elif train_config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                params_to_update,
                lr=train_config['learning_rate'],
                momentum=0.9,
                weight_decay=train_config['weight_decay']
            )
        elif train_config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                params_to_update,
                lr=train_config['learning_rate'],
                weight_decay=train_config['weight_decay']
            )
        
        # 設置學習率調度器
        if train_config['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=train_config['step_size'],
                gamma=train_config['gamma']
            )
        elif train_config['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_config['num_epochs']
            )
        elif train_config['scheduler'] == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=3,
                factor=0.5
            )
    
    def train_model(self):
        """訓練模型"""
        train_config = self.config['training']
        early_stop_config = self.config['early_stopping']
        
        print("開始訓練...")
        since = time.time()
        
        best_model_wts = copy.deepcopy(self.model.state_dict())
        
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        for epoch in range(train_config['num_epochs']):
            print(f'Epoch {epoch+1}/{train_config["num_epochs"]}')
            print('-' * 10)
            
            # 每個epoch都有訓練和驗證階段
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
                            self.optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                if phase == 'train':
                    train_losses.append(epoch_loss)
                    train_accuracies.append(epoch_acc.cpu().numpy())
                    if self.scheduler and self.config['training']['scheduler'] != 'plateau':
                        self.scheduler.step()
                else:
                    val_losses.append(epoch_loss)
                    val_accuracies.append(epoch_acc.cpu().numpy())
                    if self.scheduler and self.config['training']['scheduler'] == 'plateau':
                        self.scheduler.step(epoch_acc)
                    
                    # 檢查是否是最佳模型
                    if epoch_acc > self.best_acc:
                        self.best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        self.early_stopping_counter = 0
                    else:
                        self.early_stopping_counter += 1
            
            # 早停檢查
            if (early_stop_config['enabled'] and 
                self.early_stopping_counter >= early_stop_config['patience']):
                print(f"早停觸發！在第 {epoch+1} 輪停止訓練")
                break
            
            print()
        
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {self.best_acc:4f}')
        
        # 加載最佳模型權重
        self.model.load_state_dict(best_model_wts)
        
        # 繪製訓練曲線
        self.plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
        
        return self.model
    
    def plot_training_curves(self, train_losses, val_losses, train_accs, val_accs):
        """繪製訓練曲線"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Acc')
        plt.plot(val_accs, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        
        plt.tight_layout()
        save_path = self.config['output']['plot_save_path']
        plt.savefig(save_path)
        print(f"訓練曲線已保存到: {save_path}")
        plt.show()
    
    def save_model(self):
        """保存模型"""
        save_path = self.config['output']['model_save_path']
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'model_architecture': self.config['model']['architecture'],
            'config': self.config
        }, save_path)
        print(f"模型已保存到: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='使用配置文件訓練貓狗分類器')
    parser.add_argument('--config', type=str, default='config/training_config.yml',
                       help='訓練配置文件路徑')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"找不到配置文件: {args.config}")
        return
    
    # 創建訓練器
    trainer = ConfigurableTrainer(args.config)
    
    # 檢查數據路徑
    data_dir = trainer.config['data']['data_dir']
    if not os.path.exists(data_dir):
        print(f"錯誤：找不到數據路徑 {data_dir}")
        return
    
    # 訓練流程
    trainer.load_data()
    trainer.build_model()
    trainer.train_model()
    trainer.save_model()
    
    print("訓練完成！")

if __name__ == '__main__':
    main()
