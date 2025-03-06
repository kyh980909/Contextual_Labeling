import torch
from PIL import Image
import torchvision.transforms as T
from models import build_model

def main():
    # Device 설정 / Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 최소한의 args 객체 생성 (실제 프로젝트에 맞게 필요 인자를 채워주세요)
    # Create a minimal args object (fill necessary arguments according to your project requirements)
    class Args:
        def __init__(self):
            self.hidden_dim = 256
            self.obj_temp = 1.0
            self.num_classes = 81
            self.train_set = 'owod_t1_train.txt'
            self.test_set = 'owod_t1_test.txt'
            self.data_root = './data/OWOD'
            self.device = device
            # 추가 인자들이 필요하다면 여기 추가 (Add more parameters if needed)
    args = Args()
    
    # 모델 빌드 및 로드 / Build the model
    # 이 함수는 현재 프로젝트에서 모델, criterion, postprocessors 등을 생성합니다.
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    
    # 체크포인트 로드 (옵션) / Load checkpoint (optional)
    # checkpoint = torch.load("path/to/checkpoint.pth", map_location=device)
    # model.load_state_dict(checkpoint['model'])
    
    # 이미지 로드 및 전처리 / Load and preprocess an image
    img_path = "path/to/sample_image.jpg"  # inference할 이미지 경로
    image = Image.open(img_path).convert("RGB")
    transform = T.Compose([
        T.Resize((800, 800)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 추론 수행 / Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # 후처리 (선택 사항) / Post-processing (if needed)
    # 예를 들어, postprocessors를 사용하여 bounding box, class score 등으로 변환 가능
    print("Inference outputs:")
    print(outputs)  # raw 모델 출력 확인

if __name__ == '__main__':
    main() 