#!/usr/bin/env python3
"""
分段續訓 + 牆鐘時間保護版
- --resume 從上次 checkpoint 接續
- --save-every 每 N 個 epoch 固定存檔，且每輪都覆蓋 ckpt_<arch>_latest.pth
- --max-wall-min 逼近 6h 前自動保存並優雅退出，避免 GH Actions 被強殺
"""

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import argparse, os, time, matplotlib.pyplot as plt, numpy as np

class OverfitTrainer:
    def __init__(self, data_dir, target_accuracy=1.0):
        self.data_dir = data_dir
        self.target_accuracy = target_accuracy
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"🎯 目標訓練準確率: {target_accuracy*100:.1f}%")
        print(f"🔧 使用設備: {self.device}")

        torch.manual_seed(42); np.random.seed(42)

        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
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
        self.dataloaders, self.dataset_sizes, self.class_names = {}, {}, []
        self.arch_name = None  # ← 記錄本次架構

    def load_data(self, batch_size=8, num_workers=4):
        print("📂 正在加載數據...")
        image_datasets = {
            x: datasets.ImageFolder(os.path.join(self.data_dir, x), self.data_transforms[x])
            for x in ['train', 'val']
        }
        self.dataloaders = {
            'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=(self.device.type=="cuda"),
                                persistent_workers=(num_workers>0)),
            'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=(self.device.type=="cuda"),
                              persistent_workers=(num_workers>0))
        }
        self.dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        self.class_names = image_datasets['train'].classes
        print(f"✅ 訓練集大小: {self.dataset_sizes['train']}")
        print(f"✅ 驗證集大小: {self.dataset_sizes['val']}")
        print(f"✅ 類別: {self.class_names}")

    def build_model(self, architecture='resnet18', lr=1e-5, weight_decay=0.0):
        self.arch_name = architecture
        print(f"🏗️ 正在構建模型: {architecture} (weights=None)")
        if architecture == 'resnet50':
            self.model = models.resnet50(weights=None)
        elif architecture == 'resnet101':
            self.model = models.resnet101(weights=None)
        elif architecture == 'resnet34':
            self.model = models.resnet34(weights=None)
        else:
            self.model = models.resnet18(weights=None)

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        for p in self.model.parameters(): p.requires_grad = True
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        print("✅ 模型已構建，所有層均可訓練")

    # ---------- Checkpoint I/O ----------
    def _ckpt_name(self, kind, epoch=None):
        if kind == "latest":
            return f"ckpt_{self.arch_name}_latest.pth"
        if kind == "best":
            return f"ckpt_{self.arch_name}_best.pth"
        if kind == "epoch":
            return f"ckpt_{self.arch_name}_epoch{epoch}.pth"
        return f"ckpt_{self.arch_name}.pth"

    def save_ckpt(self, epoch, best_acc, path=None):
        if path is None:
            path = self._ckpt_name("epoch", epoch+1)
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_acc': float(best_acc),
            'class_names': self.class_names,
            'arch': self.arch_name,
        }
        torch.save(state, path)
        print(f'💾 已保存 checkpoint: {path} (epoch={epoch+1})')

    def load_ckpt(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = int(ckpt.get('epoch', -1)) + 1
        best_acc = float(ckpt.get('best_acc', 0.0))
        if ckpt.get('class_names'): self.class_names = ckpt['class_names']
        if ckpt.get('arch'): self.arch_name = ckpt['arch']
        print(f'↩️ 讀取 checkpoint: {path}，從 epoch {start_epoch+1} 繼續（best_val_acc={best_acc:.4f}）')
        return start_epoch, best_acc

    # ---------- Train ----------
    def train_to_perfection(self, max_epochs=200, resume_path='', save_every=5, max_wall_min=330):
        print(f"🚀 開始訓練到 {self.target_accuracy*100:.1f}% 準確率（最多 {max_epochs} epochs）")
        print("=" * 60)

        since = time.time()
        best_acc = 0.0
        start_epoch = 0
        if resume_path and os.path.exists(resume_path):
            start_epoch, best_acc = self.load_ckpt(resume_path)

        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
        for epoch in range(start_epoch, max_epochs):
            print(f'Epoch {epoch+1}/{max_epochs}')
            print('-' * 40)
            epoch_train_acc = 0.0

            for phase in ['train', 'val']:
                is_train = (phase == 'train')
                self.model.train(is_train)
                running_loss, running_corrects = 0.0, 0
                total_steps = len(self.dataloaders[phase])

                for bidx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    # 牆鐘時間防護
                    if max_wall_min and (time.time() - since) / 60.0 > max_wall_min:
                        self.save_ckpt(epoch, best_acc, path=self._ckpt_name("latest"))
                        with open('NEED_MORE.txt', 'w') as f: f.write('continue')
                        print('⏱️ 達到本輪時間配額，已保存 ckpt，優雅退出以避免 6h 強制中斷。')
                        return

                    inputs = inputs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    self.optimizer.zero_grad(set_to_none=True)

                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)

                    if is_train:
                        loss.backward()
                        self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels)

                    if (bidx + 1) % 20 == 0 or (bidx + 1) == total_steps:
                        done = (bidx + 1) * inputs.size(0)
                        total = self.dataset_sizes[phase]
                        print(f"  [{phase}] step {bidx+1}/{total_steps} ~{min(done, total)}/{total} samples", flush=True)

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                print(f'{phase} Loss: {epoch_loss:.6f} Acc: {epoch_acc:.4f} ({epoch_acc*100:.2f}%)')

                if is_train:
                    train_losses.append(epoch_loss)
                    train_accuracies.append(epoch_acc.item())
                    epoch_train_acc = epoch_acc
                else:
                    val_losses.append(epoch_loss)
                    val_accuracies.append(epoch_acc.item())
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        self.save_ckpt(epoch, best_acc, path=self._ckpt_name("best"))

            # 每個 epoch 結束：更新 latest + 週期固存
            self.save_ckpt(epoch, best_acc, path=self._ckpt_name("latest"))
            if (epoch + 1) % save_every == 0:
                self.save_ckpt(epoch, best_acc, path=self._ckpt_name("epoch", epoch+1))

            if epoch_train_acc >= self.target_accuracy:
                with open('TRAINING_COMPLETE.txt', 'w') as f: f.write('done')
                print(f"\n🎉 達到目標訓練準確率 {self.target_accuracy*100:.1f}%！在第 {epoch+1} 輪")
                break
            print()

        elapsed = time.time() - since
        print(f'⏱️ 訓練耗時: {elapsed // 60:.0f}m {elapsed % 60:.0f}s')
        print(f'🏅 最佳驗證準確率: {best_acc:.4f}')
        self.plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)

    def plot_training_curves(self, train_losses, val_losses, train_accs, val_accs):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1); plt.plot(train_losses,label='Train'); plt.plot(val_losses,label='Val')
        plt.title('Loss'); plt.xlabel('Epoch'); plt.grid(True); plt.legend()
        plt.subplot(1, 3, 2); plt.plot(train_accs,label='Train'); plt.plot(val_accs,label='Val')
        plt.axhline(y=self.target_accuracy, color='g', ls='--', label='Target')
        plt.title('Accuracy'); plt.xlabel('Epoch'); plt.grid(True); plt.legend()
        plt.subplot(1, 3, 3); plt.plot(train_accs, lw=2, label='Train')
        plt.axhline(y=self.target_accuracy, color='g', ls='--', label='Target')
        plt.ylim(0.9, 1.01); plt.title('Training Acc (Zoom)'); plt.grid(True); plt.legend()
        plt.tight_layout(); plt.savefig('overfit_training_curves.png', dpi=300, bbox_inches='tight')
        print("✅ 訓練曲線已保存: overfit_training_curves.png")
    
    def save_model(self, filepath='best_cat_dog_model.pth'):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'arch': self.arch_name,
            'model_architecture': self.arch_name,  # 兼容舊欄位
            'target_accuracy': self.target_accuracy,
            'training_type': 'overfitted_for_perfect_accuracy'
        }, filepath)
        print(f"🎯 模型已保存: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='分段續訓的貓狗分類器')
    parser.add_argument('--data-dir', type=str, default='file/kaggle_cats_vs_dogs_f')
    parser.add_argument('--architecture', type=str, default='resnet18',
                        choices=['resnet18','resnet34','resnet50','resnet101'])
    parser.add_argument('--target-accuracy', type=float, default=1.0)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--resume', type=str, default='', help='checkpoint 路徑（續訓）')
    parser.add_argument('--save-every', type=int, default=5, help='每 N 個 epoch 固定存 checkpoint')
    parser.add_argument('--max-wall-min', type=int, default=330, help='本輪最多訓練分鐘數（<360）')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"❌ 找不到數據路徑: {args.data_dir}")
        return

    print("🎯 100% 訓練準確率專用訓練器")
    print("=" * 50)
    trainer = OverfitTrainer(args.data_dir, args.target_accuracy)
    trainer.load_data(batch_size=args.batch_size, num_workers=args.num_workers)
    trainer.build_model(args.architecture, lr=args.lr, weight_decay=args.weight_decay)
    trainer.train_to_perfection(max_epochs=args.max_epochs,
                                resume_path=args.resume,
                                save_every=args.save_every,
                                max_wall_min=args.max_wall_min)
    if os.path.exists('TRAINING_COMPLETE.txt'):
        trainer.save_model('best_cat_dog_model.pth')
        print("\n🎉 訓練完成！你可以執行：python predict.py --model best_cat_dog_model.pth --evaluate-all")

main()
