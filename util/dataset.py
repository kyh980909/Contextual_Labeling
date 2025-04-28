import numpy as np

from datasets.torchvision_datasets.open_world import OWDetection
from datasets.coco import make_coco_transforms

def get_datasets(args):
    print(args.dataset)

    train_set = args.train_set
    test_set = args.test_set
    dataset_train = OWDetection(args, args.data_root, image_set=args.train_set, transforms=make_coco_transforms(args.train_set), dataset = args.dataset)
    dataset_val = OWDetection(args, args.data_root, image_set=args.test_set, dataset = args.dataset, transforms=make_coco_transforms(args.test_set))

    return dataset_train, dataset_val

def get_sample_from_dataloader(data_loader, index):
    """
    DataLoader 내부의 Dataset에서 특정 인덱스의 데이터를 가져오는 함수

    Args:
        data_loader: PyTorch DataLoader 객체
        index (int): 추출할 데이터 인덱스

    Returns:
        image_tensor: shape (1, 3, H, W)
        target: dict
    """
    dataset = data_loader.dataset
    image, target = dataset[index]

    if isinstance(image, np.ndarray):
        image = to_tensor(image)

    if image.ndim == 3:
        image = image.unsqueeze(0)  # [3, H, W] → [1, 3, H, W]

    return image, target