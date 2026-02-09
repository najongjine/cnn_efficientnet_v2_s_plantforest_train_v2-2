import os
import shutil
import json
import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# -----------------------------------------------------------
# 0. 설정
# -----------------------------------------------------------
USE_FILTERED_CLASSES = False
ALLOWED_CLASSES = ["Healthy Wheat", "Leaf Blight", "Stem fly"]
SPLIT_FOLDERS = ["train", "validation"]

# ✅ [수정됨] 로컬 저장 경로 설정 (현재 폴더 기준 'models' 폴더에 저장)
BASE_DIR = os.getcwd()
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "my_models")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

CKPT_PATH = os.path.join(MODEL_SAVE_DIR, "efficientnet_v2_s_plantforestdisease.pt")
CLASS_PATH = os.path.join(MODEL_SAVE_DIR, "efficientnet_v2_s_plantforestdisease.json")

# ✅ [수정됨] 배치 사이즈 조절 (로컬 GPU 메모리에 따라 조절하세요. OOM 에러나면 8로 줄이세요)
BATCH_SIZE = 32   
NUM_WORKERS = 0   # 윈도우 로컬에서는 멀티프로세싱 오류 방지를 위해 0으로 설정하는 것이 안전합니다.
EPOCHS = 10       
LR = 1e-4
WEIGHT_DECAY = 1e-4
IMG_SIZE = 384

# -----------------------------------------------------------
# 1. 데이터셋 다운로드 및 준비
# -----------------------------------------------------------
destination_path = os.path.join(BASE_DIR, "datasets")

# 1. 데이터셋 폴더가 없으면 -> 다운로드 받고 이동시킴
if not os.path.exists(destination_path):
    print("📂 데이터셋 폴더가 없습니다. 다운로드를 시작합니다 (Kaggle Hub)...")
    try:
        # 다운로드를 이 안에서 수행
        path = kagglehub.dataset_download("freedomfighter1290/wheat-disease")
        print(f"\n✅ 다운로드 경로: {path}")
        
        # Wheat_Disease 하위 폴더 처리
        path_with_subfolder = os.path.join(path, 'Wheat_Disease')
        if os.path.exists(path_with_subfolder):
            path = path_with_subfolder

        # 필터링 로직 (USE_FILTERED_CLASSES가 True일 때만)
        if USE_FILTERED_CLASSES:
            for split in SPLIT_FOLDERS:
                split_path = os.path.join(path, split)
                if os.path.exists(split_path):
                    for item in os.listdir(split_path):
                        item_path = os.path.join(split_path, item)
                        if os.path.isdir(item_path) and item not in ALLOWED_CLASSES:
                            shutil.rmtree(item_path)

        # 폴더 이동
        print("📦 데이터셋 폴더 이동 및 정리 중...")
        shutil.move(path, destination_path)
        print(f"✅ 데이터셋 준비 완료: {destination_path}")

    except Exception as e:
        print(f"❌ 다운로드 또는 이동 실패: {e}")
        print("💡 팁: 'kaggle.json' 파일 확인 또는 인터넷 연결을 확인하세요.")
        exit()

# 2. 데이터셋 폴더가 이미 있으면 -> 다운로드 아예 안 함
else:
    print(f"✅ 기존 데이터셋 폴더를 발견했습니다. 다운로드를 건너뜁니다: {destination_path}")


# -----------------------------------------------------------
# 2. 전처리 및 DataLoader
# -----------------------------------------------------------
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_tf = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_ds = datasets.ImageFolder(root=os.path.join(destination_path, "train"), transform=train_tf)
val_ds   = datasets.ImageFolder(root=os.path.join(destination_path, "validation"), transform=val_tf)

# Windows에서는 num_workers=0 권장 (오류 발생 시)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# 클래스 정보 저장
class_names = train_ds.classes
with open(CLASS_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False)

print(f"클래스 목록: {class_names}")

# -----------------------------------------------------------
# 3. 모델 정의 및 학습
# -----------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")

weights = models.EfficientNet_V2_S_Weights.DEFAULT
model = models.efficientnet_v2_s(weights=weights)

# Classifier 수정
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_acc = 0.0
print("🚀 학습 시작...")

for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss, total = 0, 0
    train_correct = 0  # ✅ [추가] 훈련 정답 개수 초기화

    for batch_idx, (x, y) in enumerate(train_dl):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # 모델 예측값 한 번만 계산해서 변수에 저장
        outputs = model(x) 
        loss = criterion(outputs, y)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        total += y.size(0)
        
        # ✅ [추가] 훈련 정확도 계산 로직
        preds = outputs.argmax(dim=1)
        train_correct += (preds == y).sum().item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(train_dl)}] Loss: {loss.item():.4f}", end='\r')

    scheduler.step()

    # 훈련 정확도 계산
    train_acc = train_correct / total * 100 # ✅ [추가]

    # 검증 (Validation)
    model.eval()
    correct, val_total = 0, 0 # (변수명 겹치지 않게 주의)
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item() # 여기는 검증 정답 개수
            val_total += y.size(0)

    val_acc = correct / val_total * 100
    
    # ✅ [수정] 출력문에 Train Acc 추가
    print(f"\n[{epoch}/{EPOCHS}] Train Loss: {total_loss/total:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            "state_dict": model.state_dict(),
            "class_names": class_names,
            "model_type": "efficientnet_v2_s"
        }, CKPT_PATH)
        print(f"  🎉 모델 저장됨: {CKPT_PATH}")

print(f"\n✅ 최종 학습 완료. 최고 정확도: {best_acc:.2f}%")