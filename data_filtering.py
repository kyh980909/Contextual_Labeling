import os
import glob
import xml.etree.ElementTree as ET

# 경로 설정
img_dir = './data/OWOD/JPEGImages'
ann_dir = './data/OWOD/Annotations'
test_list_path = './data/OWOD/ImageSets/TOWOD/owod_all_task_test.txt'
save_txt_path = './data/MyDataset/filtered_test.txt'

# 1. 테스트 ID 리스트 불러오기
with open(test_list_path, 'r') as f:
    test_ids = set(line.strip() for line in f if line.strip())

# 2. 이미지-어노테이션 매칭 후 필터링
filtered_ids = []
for xml_path in glob.glob(os.path.join(ann_dir, '*.xml')):
    base_name = os.path.splitext(os.path.basename(xml_path))[0]
    if base_name in test_ids:
        # 객체 수 세기
        tree = ET.parse(xml_path)
        root = tree.getroot()
        object_count = len(root.findall('object'))

        # 조건: 객체 수가 N개 이하인 경우만 필터링
        N = 3  # 원하는 최대 객체 수
        if object_count <= N:
            filtered_ids.append(base_name)

# 3. 결과 저장
os.makedirs(os.path.dirname(save_txt_path), exist_ok=True)
with open(save_txt_path, 'w') as f:
    for img_id in sorted(filtered_ids):
        f.write(f"{img_id}\n")

print(f"✅ 필터링 완료: {len(filtered_ids)}개 이미지 저장됨 → {save_txt_path}")