import cv2
import numpy as np
import torch

def mask_known_regions(image: np.ndarray, boxes: np.ndarray, mask_type='gray') -> np.ndarray:
    """
    Args:
        image (np.ndarray): (H, W, 3) BGR 이미지
        boxes (np.ndarray): (N, 4) – [x1, y1, x2, y2]
        mask_type (str): 'gray', 'black', 'blur'

    Returns:
        np.ndarray: 마스킹된 이미지
    """
    masked_image = image.copy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        roi = masked_image[y1:y2, x1:x2]

        if mask_type == 'gray':
            masked_image[y1:y2, x1:x2] = 128  # 중간 회색
        elif mask_type == 'black':
            masked_image[y1:y2, x1:x2] = 0
        elif mask_type == 'blur':
            masked_image[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (23, 23), 30)
        else:
            raise ValueError(f"Unknown mask_type: {mask_type}")
    return masked_image
