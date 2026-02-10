import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# 0. 설정 (학습 코드와 동일하게 유지)
# -----------------------------------------------------------
BASE_DIR = os.getcwd()
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "my_models")
CKPT_PATH = os.path.join(MODEL_SAVE_DIR, "efficientnet_v2_s_plantforestdisease.pt")

IMG_SIZE = 384
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 분석할 이미지 경로 (여기를 수정하세요)
TEST_IMAGE_PATH = r"C:\Users\najon\OneDrive\사진\plant4.png" 

# -----------------------------------------------------------
# 1. 모델 로드 및 Grad-CAM 클래스 정의
# -----------------------------------------------------------
def load_trained_model(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"❌ 모델 파일이 없습니다: {ckpt_path}")

    print(f"📂 모델 로딩 중... ({ckpt_path})")
    checkpoint = torch.load(ckpt_path, map_location=device)
    class_names = checkpoint['class_names']
    
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(class_names))
    model.load_state_dict(checkpoint['state_dict'])
    
    model.to(device)
    model.eval()
    return model, class_names

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook 등록
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # 1. Forward Pass
        outputs = self.model(x)
        if class_idx is None:
            class_idx = outputs.argmax(dim=1).item()
        
        # 2. Backward Pass (Gradients 계산)
        self.model.zero_grad()
        score = outputs[0, class_idx]
        score.backward()
        
        # 3. Grad-CAM 계산
        gradients = self.gradients[0]   # (C, H, W)
        activations = self.activations[0] # (C, H, W)
        
        # Global Average Pooling (가중치 계산)
        weights = torch.mean(gradients, dim=(1, 2))
        
        # 가중치와 Feature Map 결합
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=DEVICE)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # ReLU 적용
        cam = F.relu(cam)
        
        # 정규화 (0~1)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().detach().numpy(), class_idx, outputs

# -----------------------------------------------------------
# 2. 시각화 함수들
# -----------------------------------------------------------
def show_cam_on_image(img_path, mask, class_name, confidence):
    img = np.array(Image.open(img_path).convert("RGB"))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.float32(img) / 255
    
    # Heatmap 생성
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1] # BGR to RGB
    
    # 원본 이미지와 겹치기
    cam = heatmap * 0.4 + img * 0.6
    cam = cam / np.max(cam)
    
    plt.figure(figsize=(12, 5))
    
    # 원본
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    # Grad-CAM 결과
    plt.subplot(1, 2, 2)
    plt.imshow(np.uint8(255 * cam))
    plt.title(f"Grad-CAM\nPred: {class_name} ({confidence:.1f}%)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_feature_maps(model, img_tensor):
    # 첫 번째 합성곱 층 (Stem)의 Feature Map 가져오기
    # EfficientNet V2 S의 구조상 features[0]이 첫 Conv 층입니다.
    with torch.no_grad():
        features = model.features[0](img_tensor)
    
    features = features[0].cpu().numpy() # (Batch, Channel, H, W) -> (Channel, H, W)
    
    # 채널 중 앞 16개만 시각화
    num_channels = min(16, features.shape[0])
    
    plt.figure(figsize=(16, 8))
    for i in range(num_channels):
        plt.subplot(2, 8, i + 1)
        plt.imshow(features[i], cmap='viridis')
        plt.axis('off')
        plt.title(f"Ch {i}")
    
    plt.suptitle(f"Feature Maps (First Layer): {num_channels} channels", fontsize=16)
    plt.show()

# -----------------------------------------------------------
# 3. 메인 실행
# -----------------------------------------------------------
if __name__ == "__main__":
    # 이미지 전처리
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if os.path.exists(TEST_IMAGE_PATH):
        # 모델 로드
        model, class_names = load_trained_model(CKPT_PATH, DEVICE)
        
        # 이미지 로드
        image = Image.open(TEST_IMAGE_PATH)
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        print("\n🔍 [1] Feature Map 시각화 (Low-level Features)...")
        visualize_feature_maps(model, input_tensor)
        
        print("\n🔍 [2] Grad-CAM Heatmap 시각화 (Decision Regions)...")
        # EfficientNet V2의 마지막 Conv 층: features[-1]
        target_layer = model.features[-1]
        grad_cam = GradCAM(model, target_layer)
        
        mask, class_idx, outputs = grad_cam(input_tensor)
        
        # 마스크 크기를 원본 이미지 크기로 맞춤
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        
        # 확률 계산
        probs = F.softmax(outputs, dim=1)
        confidence = probs[0][class_idx].item() * 100
        pred_class = class_names[class_idx]
        
        show_cam_on_image(TEST_IMAGE_PATH, mask, pred_class, confidence)
        
    else:
        print(f"⚠️ 이미지를 찾을 수 없습니다: {TEST_IMAGE_PATH}")
        print("테스트할 이미지 경로를 확인해주세요.")