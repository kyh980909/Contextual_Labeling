import cv2
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

def draw_boxes(image: np.ndarray,
               scores: np.ndarray,
               labels: np.ndarray,
               boxes: np.ndarray,
               class_names: list = None,
               thickness=2,
               font_scale=0.5) -> np.ndarray:
    """
    Args:
        image (np.ndarray): BGR 이미지
        boxes (np.ndarray): [N, 4] – x1y1x2y2
        labels (list): 각 박스에 대응되는 클래스 이름 문자열 리스트
    """
    cmap = plt.get_cmap("tab20", 20)
    label_color_map = {}

    fig, ax = plt.subplots(1, 1)
    ax.imshow(image)

    for score, label, box in zip(scores, labels, boxes):
        cls_id = label.item()
        x1, y1, x2, y2 = map(int, box)

        if class_names is not None:
            if cls_id not in label_color_map:
                idx = len(label_color_map) % 20
                rgb = cmap(idx)[:3]
                label_color_map[cls_id] = rgb
            box_color = label_color_map[cls_id]
        else:
            if cls_id not in label_color_map:
                idx = len(label_color_map) % 20
                rgb = cmap(idx)[:3]
                label_color_map[cls_id] = rgb
            box_color = label_color_map[cls_id]

        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=thickness,
                             edgecolor=box_color, facecolor='none')
        ax.add_patch(rect)

        label_name = class_names[cls_id] if class_names else f"class {cls_id}"
        label_text = f"{label_name}: {score:.2f}"
        ax.text(x1, y1 - 5, label_text, fontsize=10,
                bbox=dict(facecolor=box_color, alpha=0.6, edgecolor='none'),
                color='white')

    ax.axis('off')
    plt.tight_layout()
    plt.show()
    return image

def tensor_to_img(image: torch.Tensor) -> np.ndarray:
    inv_normalize = T.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )

    image = inv_normalize(image)[0]
    image = image.permute(1, 2, 0).numpy()
    return image