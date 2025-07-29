import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse
import os

class CatDogPredictor:
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
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)  # 2個類別
        
        # 加載權重
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        self.class_names = checkpoint['class_names']
        print(f"模型加載成功！類別: {self.class_names}")
        
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
            return
        
        print(f"找到 {len(image_files)} 張圖片，開始預測...")
        
        results = []
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            try:
                predicted_class, confidence, probabilities = self.predict_single_image(image_path)
                results.append({
                    'filename': image_file,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'cat_probability': probabilities[0] if self.class_names[0] == 'cat' else probabilities[1],
                    'dog_probability': probabilities[1] if self.class_names[1] == 'dog' else probabilities[0]
                })
                print(f"{image_file}: {predicted_class} (信心度: {confidence:.3f})")
            except Exception as e:
                print(f"預測 {image_file} 時出錯: {e}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='貓狗分類器預測工具')
    parser.add_argument('--model', type=str, default='best_cat_dog_model.pth',
                       help='訓練好的模型路徑')
    parser.add_argument('--image', type=str, help='要預測的單張圖片路徑')
    parser.add_argument('--folder', type=str, help='要預測的圖片資料夾路徑')
    
    args = parser.parse_args()
    
    if not args.image and not args.folder:
        print("請指定 --image 或 --folder 參數")
        return
    
    # 檢查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"找不到模型文件: {args.model}")
        print("請先訓練模型或檢查模型路徑")
        return
    
    # 創建預測器
    predictor = CatDogPredictor(args.model)
    
    if args.image:
        # 預測單張圖片
        try:
            predicted_class, confidence, probabilities = predictor.predict_single_image(args.image)
            print(f"\n預測結果:")
            print(f"圖片: {args.image}")
            print(f"預測類別: {predicted_class}")
            print(f"信心度: {confidence:.3f}")
            print(f"詳細概率:")
            for i, class_name in enumerate(predictor.class_names):
                print(f"  {class_name}: {probabilities[i]:.3f}")
        except Exception as e:
            print(f"預測失敗: {e}")
    
    elif args.folder:
        # 批次預測
        try:
            results = predictor.predict_batch_images(args.folder)
            if results:
                print(f"\n批次預測完成，共處理 {len(results)} 張圖片")
                
                # 統計結果
                cat_count = sum(1 for r in results if r['predicted_class'] == 'cat')
                dog_count = sum(1 for r in results if r['predicted_class'] == 'dog')
                avg_confidence = sum(r['confidence'] for r in results) / len(results)
                
                print(f"\n統計結果:")
                print(f"預測為貓的圖片: {cat_count}")
                print(f"預測為狗的圖片: {dog_count}")
                print(f"平均信心度: {avg_confidence:.3f}")
        except Exception as e:
            print(f"批次預測失敗: {e}")

if __name__ == '__main__':
    main()
