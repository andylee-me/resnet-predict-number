import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms, models
import os
import time
import copy
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class CatDogClassifier:
    def __init__(self, data_dir, batch_size=32, learning_rate=0.001, num_epochs=25):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 數據預處理
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.dataloaders = {}
        self.dataset_sizes = {}
        self.class_names = []
        
    def load_data(self):
        """加載和預處理數據"""
        print("正在加載數據...")
        
        # 創建數據集
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x),
                                                self.data_transforms[x])
                         for x in ['train', 'val']}
        
        # 創建數據加載器
        self.dataloaders = {x: DataLoader(image_datasets[x], batch_size=self.batch_size,
                                        shuffle=True, num_workers=4)
                          for x in ['train', 'val']}
        
        self.dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        self.class_names = image_datasets['train'].classes
        
        print(f"訓練集大小: {self.dataset_sizes['train']}")
        print(f"驗證集大小: {self.dataset_sizes['val']}")
        print(f"類別: {self.class_names}")
        
    def build_model(self):
        """構建模型（使用預訓練的ResNet18）"""
        print("正在構建模型...")
        
        # 加載預訓練的ResNet18
        self.model = models.resnet18(pretrained=True)
        
        # 凍結特徵提取層
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 替換最後的全連接層
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)  # 2個類別：貓和狗
        
        self.model = self.model.to(self.device)
        
        # 只優化最後一層的參數
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=self.learning_rate)
        
    def train_model(self):
        """訓練模型"""
        print("開始訓練...")
        
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch}/{self.num_epochs - 1}')
            print('-' * 10)
            
            # 每個epoch都有訓練和驗證階段
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # 設置模型為訓練模式
                else:
                    self.model.eval()   # 設置模型為評估模式
                
                running_loss = 0.0
                running_corrects = 0
                
                # 遍歷數據
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    # 零梯度
                    self.optimizer.zero_grad()
                    
                    # 前向傳播
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        
                        # 反向傳播和優化（僅在訓練階段）
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                    
                    # 統計
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                # 記錄損失和準確率
                if phase == 'train':
                    train_losses.append(epoch_loss)
                    train_accuracies.append(epoch_acc.cpu().numpy())
                else:
                    val_losses.append(epoch_loss)
                    val_accuracies.append(epoch_acc.cpu().numpy())
                
                # 深拷貝模型（如果是最佳模型）
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
            
            print()
        
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')
        
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
        plt.savefig('training_curves.png')
        plt.show()
    
    def save_model(self, filepath='cat_dog_classifier.pth'):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'model_architecture': 'resnet18'
        }, filepath)
        print(f"模型已保存到 {filepath}")
    
    def predict_image(self, image_path):
        """預測單張圖片"""
        self.model.eval()
        
        # 加載和預處理圖片
        image = Image.open(image_path).convert('RGB')
        transform = self.data_transforms['val']
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # 預測
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            _, predicted = torch.max(outputs, 1)
        
        # 返回結果
        predicted_class = self.class_names[predicted.item()]
        confidence = probabilities[predicted.item()].item()
        
        return predicted_class, confidence

def main():
    # 設置數據路徑
    data_dir = 'file/kaggle_cats_vs_dogs_f'  # 你的數據集路徑
    
    # 檢查數據路徑是否存在
    if not os.path.exists(data_dir):
        print(f"錯誤：找不到數據路徑 {data_dir}")
        print("請確保數據集已下載並放在正確的位置")
        return
    
    # 創建分類器實例
    classifier = CatDogClassifier(
        data_dir=data_dir,
        batch_size=32,
        learning_rate=0.001,
        num_epochs=25
    )
    
    # 加載數據
    classifier.load_data()
    
    # 構建模型
    classifier.build_model()
    
    # 訓練模型
    trained_model = classifier.train_model()
    
    # 保存模型
    classifier.save_model('best_cat_dog_model.pth')
    
    print("訓練完成！")

if __name__ == '__main__':
    main()
