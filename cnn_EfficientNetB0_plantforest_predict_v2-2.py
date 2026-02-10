import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F

# -----------------------------------------------------------
# 1. 설정 (학습 코드와 동일하게 유지)
# -----------------------------------------------------------
BASE_DIR = os.getcwd()
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "my_models")
CKPT_PATH = os.path.join(MODEL_SAVE_DIR, "efficientnet_v2_s_plantforestdisease.pt")

IMG_SIZE = 384
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 테스트할 이미지 경로 (여기를 수정해서 사용하세요)
# 예: "datasets/test/leaf_blight_01.jpg"
TEST_IMAGE_PATH = r"C:\Users\najon\OneDrive\사진\plant4.png" 

# -----------------------------------------------------------
# 2. 모델 불러오기 함수
# -----------------------------------------------------------
def load_trained_model(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"❌ 모델 파일이 없습니다: {ckpt_path}")

    print(f"📂 모델 로딩 중... ({ckpt_path})")
    
    # 1. 체크포인트 파일 로드
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 2. 클래스 정보 가져오기
    class_names = checkpoint['class_names']
    num_classes = len(class_names)
    
    # 3. 모델 구조 생성 (EfficientNet V2 S)
    # weights=None으로 설정 (저장된 가중치를 덮어씌울 것이므로)
    model = models.efficientnet_v2_s(weights=None)
    
    # 4. 분류기(Classifier) 레이어 수정 (학습 코드와 동일하게 맞춰야 함)
    # EfficientNet V2의 classifier[1]은 마지막 Linear 레이어입니다.
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    # 5. 학습된 가중치(State Dict) 적용
    model.load_state_dict(checkpoint['state_dict'])
    
    model.to(device)
    model.eval() # 평가 모드로 전환 (Dropout, Batchnorm 등 고정)
    
    return model, class_names

# -----------------------------------------------------------
# 3. 이미지 전처리 함수 (학습 시 Validation 변환과 동일)
# -----------------------------------------------------------
def process_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ 이미지 파일이 없습니다: {image_path}")

    # 학습 코드의 val_tf와 동일한 정규화 값
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")), # 흑백 이미지 등 대비
        transforms.Resize((IMG_SIZE, IMG_SIZE)),       # 384x384 리사이즈
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    image = Image.open(image_path)
    image_tensor = transform(image)
    
    # 배치 차원 추가 (3, 384, 384) -> (1, 3, 384, 384)
    image_tensor = image_tensor.unsqueeze(0) 
    
    return image_tensor

# -----------------------------------------------------------
# 4. 실행 (메인 로직)
# -----------------------------------------------------------
if __name__ == "__main__":
    try:
        # 1. 모델 로드
        model, class_names = load_trained_model(CKPT_PATH, DEVICE)
        print(f"✅ 모델 로드 완료 (클래스 개수: {len(class_names)})")
        print(f"📋 클래스 목록: {class_names}")

        # 2. 이미지 파일 존재 여부 확인 (테스트용 더미 파일 생성 방지)
        if not os.path.exists(TEST_IMAGE_PATH):
            print(f"\n⚠️ 주의: '{TEST_IMAGE_PATH}' 파일이 현재 폴더에 없습니다.")
            print("테스트하려는 이미지의 정확한 경로를 'TEST_IMAGE_PATH' 변수에 입력해주세요.")
        else:
            # 3. 예측 수행
            input_tensor = process_image(TEST_IMAGE_PATH).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                # 확률 계산 (Softmax)
                probs = F.softmax(outputs, dim=1)
                
                # 가장 높은 확률의 클래스 인덱스와 확률값 추출
                top_p, top_class = probs.topk(1, dim=1)
                
                prediction = class_names[top_class.item()]
                probability = top_p.item() * 100

            # 4. 결과 출력
            print("\n" + "="*30)
            print(f"🖼️  이미지: {TEST_IMAGE_PATH}")
            print(f"🔍 예측 결과: {prediction}")
            print(f"📊 확신도(Confidence): {probability:.2f}%")
            print("="*30)

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")