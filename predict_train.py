import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse
import os
import time
from collections import defaultdict

class EnhancedCatDogPredictor:
    def __init__(self, model_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 數據預處理（與訓練時相同的驗證預處理）
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 加載模型
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path):
        """加載訓練好的模型"""
        print(f"正在加載模型: {model_path}")
        
        # 加載checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 創建模型架構
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)  # 2個類別
        
        # 加載權重
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        self.class_names = checkpoint['class_names']
        print(f"✅ 模型加載成功！類別: {self.class_names}")
        print(f"🔧 使用設備: {self.device}")
        
        return model
    
    def predict_single_image(self, image_path):
        """預測單張圖片"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"找不到圖片: {image_path}")
        
        # 加載和預處理圖片
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"無法加載圖片 {image_path}: {e}")
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 進行預測
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
        
        predicted_class = self.class_names[predicted.item()]
        confidence_score = confidence.item()
        
        return predicted_class, confidence_score, probabilities.cpu().numpy()
    
    def get_true_label_from_path(self, image_path):
        """從文件路徑推斷真實標籤"""
        path_lower = image_path.lower()
        
        # 檢查路徑中是否包含類別信息
        if '/cat/' in path_lower or '\\cat\\' in path_lower:
            return 'cat'
        elif '/dog/' in path_lower or '\\dog\\' in path_lower:
            return 'dog'
        
        # 檢查文件名
        filename = os.path.basename(image_path).lower()
        if filename.startswith('cat'):
            return 'cat'
        elif filename.startswith('dog'):
            return 'dog'
        
        return None  # 無法確定真實標籤
    
    def evaluate_dataset(self, dataset_path, dataset_name="dataset"):
        """評估整個數據集"""
        print(f"\n🔍 開始評估 {dataset_name} 數據集: {dataset_path}")
        print("=" * 60)
        
        if not os.path.exists(dataset_path):
            print(f"❌ 找不到數據集路徑: {dataset_path}")
            return None
        
        # 統計變量
        results = {
            'total': 0,
            'correct': 0,
            'misclassified': [],
            'by_class': defaultdict(lambda: {'total': 0, 'correct': 0, 'misclassified': []}),
            'predictions': {'cat': 0, 'dog': 0}
        }
        
        # 支援的圖片格式
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        
        # 遞歸搜索所有圖片
        image_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(supported_formats):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            print(f"❌ 在 {dataset_path} 中沒有找到圖片文件")
            return None
        
        print(f"📊 找到 {len(image_files)} 張圖片，開始預測...")
        
        # 進度追蹤
        start_time = time.time()
        
        for i, image_path in enumerate(image_files):
            try:
                # 獲取真實標籤
                true_label = self.get_true_label_from_path(image_path)
                
                # 進行預測
                predicted_class, confidence, probabilities = self.predict_single_image(image_path)
                
                # 更新統計
                results['total'] += 1
                results['predictions'][predicted_class] += 1
                
                if true_label:
                    results['by_class'][true_label]['total'] += 1
                    
                    # 檢查是否預測正確
                    if predicted_class == true_label:
                        results['correct'] += 1
                        results['by_class'][true_label]['correct'] += 1
                    else:
                        # 記錄錯誤分類
                        error_info = {
                            'file_path': image_path,
                            'true_label': true_label,
                            'predicted_label': predicted_class,
                            'confidence': confidence
                        }
                        results['misclassified'].append(error_info)
                        results['by_class'][true_label]['misclassified'].append(error_info)
                
                # 顯示進度
                if (i + 1) % 100 == 0 or (i + 1) == len(image_files):
                    elapsed = time.time() - start_time
                    progress = (i + 1) / len(image_files) * 100
                    print(f"⏳ 進度: {i+1}/{len(image_files)} ({progress:.1f}%) - 耗時: {elapsed:.1f}s")
                    
            except Exception as e:
                print(f"❌ 處理 {image_path} 時出錯: {e}")
        
        # 計算準確率
        if results['total'] > 0:
            accuracy = results['correct'] / results['total'] * 100
        else:
            accuracy = 0
        
        # 顯示結果
        print(f"\n📋 {dataset_name} 集評估結果:")
        print(f"{'='*50}")
        print(f"✅ {dataset_name} 集: 共 {results['total']} 張, 錯誤 {len(results['misclassified'])}, 準確率 {accuracy:.2f}%")
        
        # 按類別統計
        for class_name in self.class_names:
            if class_name in results['by_class']:
                class_stats = results['by_class'][class_name]
                class_accuracy = (class_stats['correct'] / class_stats['total'] * 100) if class_stats['total'] > 0 else 0
                print(f"   📊 {class_name}: {class_stats['total']} 張, 正確 {class_stats['correct']}, 準確率 {class_accuracy:.2f}%")
        
        # 預測分布
        print(f"\n🎯 預測分布:")
        for class_name in self.class_names:
            count = results['predictions'][class_name]
            percentage = (count / results['total'] * 100) if results['total'] > 0 else 0
            print(f"   {class_name}: {count} 張 ({percentage:.1f}%)")
        
        # 顯示錯誤分類的文件（限制顯示數量避免過多輸出）
        if results['misclassified']:
            print(f"\n❌ 錯誤分類詳情 (顯示前10個):")
            for i, error in enumerate(results['misclassified'][:10]):
                print(f"   Misclassified {i+1}: {error['file_path']}")
                print(f"      真實: {error['true_label']} → 預測: {error['predicted_label']} (信心度: {error['confidence']:.3f})")
            
            if len(results['misclassified']) > 10:
                print(f"   ... 還有 {len(results['misclassified']) - 10} 個錯誤分類")
        
        return results
    
    def predict_batch_images(self, image_folder):
        """批次預測資料夾中的圖片"""
        if not os.path.exists(image_folder):
            raise FileNotFoundError(f"找不到資料夾: {image_folder}")
        
        # 支援的圖片格式
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        
        # 獲取所有圖片文件
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(supported_formats)]
        
        if not image_files:
            print("資料夾中沒有找到支援的圖片文件")
            return []
        
        print(f"找到 {len(image_files)} 張圖片，開始預測...")
        
        results = []
        correct_predictions = 0
        total_with_labels = 0
        
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            try:
                predicted_class, confidence, probabilities = self.predict_single_image(image_path)
                
                # 嘗試從文件名判斷真實類別
                true_class = self.get_true_label_from_path(image_path)
                
                is_correct = (true_class == predicted_class) if true_class else None
                if is_correct is True:
                    correct_predictions += 1
                if true_class:
                    total_with_labels += 1
                
                results.append({
                    'filename': image_file,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'true_class': true_class,
                    'correct': is_correct,
                    'cat_probability': probabilities[0] if self.class_names[0] == 'cat' else probabilities[1],
                    'dog_probability': probabilities[1] if self.class_names[1] == 'dog' else probabilities[0]
                })
                
                status = "✅" if is_correct else "❌" if is_correct is False else "❓"
                print(f"{status} {image_file}: {predicted_class} (信心度: {confidence:.3f})")
                
            except Exception as e:
                print(f"❌ 預測 {image_file} 時出錯: {e}")
        
        # 顯示統計
        if results:
            if total_with_labels > 0:
                accuracy = correct_predictions / total_with_labels * 100
                print(f"\n📊 批次預測統計:")
                print(f"✅ 總計: 共 {len(results)} 張, 錯誤 {total_with_labels - correct_predictions}, 準確率 {accuracy:.2f}%")
            
            avg_confidence = sum(r['confidence'] for r in results) / len(results)
            cat_predictions = sum(1 for r in results if r['predicted_class'] == 'cat')
            dog_predictions = sum(1 for r in results if r['predicted_class'] == 'dog')
            
            print(f"📈 平均信心度: {avg_confidence:.3f}")
            print(f"🐱 預測為貓的圖片: {cat_predictions}")
            print(f"🐶 預測為狗的圖片: {dog_predictions}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='增強版貓狗分類器預測工具')
    parser.add_argument('--model', type=str, default='best_cat_dog_model.pth',
                       help='訓練好的模型路徑')
    parser.add_argument('--image', type=str, help='要預測的單張圖片路徑')
    parser.add_argument('--folder', type=str, help='要預測的圖片資料夾路徑')
    parser.add_argument('--evaluate-train', action='store_true',
                       help='評估訓練數據集')
    parser.add_argument('--evaluate-val', action='store_true',
                       help='評估驗證數據集')
    parser.add_argument('--evaluate-all', action='store_true',
                       help='評估所有數據集（訓練+驗證）')
    parser.add_argument('--dataset-path', type=str, default='kaggle_cats_vs_dogs_f',
                       help='數據集根目錄路徑')
    
    args = parser.parse_args()
    
    # 檢查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"找不到模型文件: {args.model}")
        print("請先訓練模型或檢查模型路徑")
        return
    
    # 創建預測器
    predictor = EnhancedCatDogPredictor(args.model)
    
    # 評估數據集
    if args.evaluate_all or args.evaluate_train:
        train_path = os.path.join(args.dataset_path, 'train')
        if os.path.exists(train_path):
            predictor.evaluate_dataset(train_path, "train")
        else:
            print(f"❌ 找不到訓練數據集: {train_path}")


  
    if args.evaluate_all or args.evaluate_val:
        train_path = os.path.join(args.dataset_path, 'train')
        if os.path.exists(train_path):
            print("⚠️ 注意：val 評估實際上使用的是 train 資料")
            predictor.evaluate_dataset(train_path, "val")
        else:
            print(f"❌ 找不到訓練數據集（用作 val）: {train_path}")

    if args.evaluate_all or args.evaluate_val:
        val_path = os.path.join(args.dataset_path, 'val')
        if os.path.exists(val_path):
            predictor.evaluate_dataset(val_path, "val")
        else:
            print(f"❌ 找不到驗證數據集: {val_path}")
    
    # 單張圖片預測
    if args.image:
        try:
            predicted_class, confidence, probabilities = predictor.predict_single_image(args.image)
            print(f"\n🎯 單張圖片預測結果:")
            print(f"圖片: {args.image}")
            print(f"預測類別: {predicted_class}")
            print(f"信心度: {confidence:.3f}")
            print(f"詳細概率:")
            for i, class_name in enumerate(predictor.class_names):
                print(f"  {class_name}: {probabilities[i]:.3f}")
        except Exception as e:
            print(f"預測失敗: {e}")
    
    # 批次預測
    elif args.folder:
        try:
            results = predictor.predict_batch_images(args.folder)
        except Exception as e:
            print(f"批次預測失敗: {e}")
    
    # 如果沒有指定任何操作，顯示幫助
    elif not (args.evaluate_all or args.evaluate_train or args.evaluate_val):
        print("請指定操作:")
        print("  --image <path>        預測單張圖片")
        print("  --folder <path>       預測資料夾中的圖片")
        print("  --evaluate-train      評估訓練數據集")
        print("  --evaluate-val        評估驗證數據集")
        print("  --evaluate-all        評估所有數據集")
        print("\n範例:")
        print("  python predict.py --evaluate-train")
        print("  python predict.py --evaluate-all")
        print("  python predict.py --image cat.jpg")


main()



