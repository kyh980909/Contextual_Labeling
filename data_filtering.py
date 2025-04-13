import os
import json
import random

def filter_data(json_file, output_file, num_objects):
    """
    Filters a COCO-style JSON annotation file to include only images with a specific number of objects.

    Args:
        json_file (str): Path to the input COCO-style JSON annotation file.
        output_file (str): Path to save the filtered JSON annotation file.
        num_objects (int): The desired number of objects per image.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    filtered_images = []
    filtered_annotations = []
    image_id_mapping = {}  # To map old image IDs to new ones
    new_image_id_counter = 1
    annotation_id_counter = 1

    # Create a dictionary to count annotations per image
    image_annotation_counts = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotation_counts:
            image_annotation_counts[image_id] = 0
        image_annotation_counts[image_id] += 1

    # Filter images and annotations
    for img in data['images']:
        image_id = img['id']
        if image_id in image_annotation_counts and image_annotation_counts[image_id] == num_objects:
            # Map old image ID to new image ID
            image_id_mapping[image_id] = new_image_id_counter
            img['id'] = new_image_id_counter
            filtered_images.append(img)
            new_image_id_counter += 1

    for ann in data['annotations']:
        if ann['image_id'] in image_id_mapping:
            ann['image_id'] = image_id_mapping[ann['image_id']]
            ann['id'] = annotation_id_counter
            filtered_annotations.append(ann)
            annotation_id_counter += 1

    # Create the filtered JSON structure
    filtered_data = {
        'info': data['info'],
        'licenses': data['licenses'],
