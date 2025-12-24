import xml.etree.ElementTree as ET
import os
import shutil
from pathlib import Path
import yaml

# ------------------------------------------------------------
#  XML to YOLO format converter for YOLOv3/YOLOv5/YOLOv8
# ------------------------------------------------------------
def convert_xml_to_yolo(xml_path, img_width, img_height, output_path, class_mapping):
    """
    Convert a single XML annotation to YOLO format.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    with open(output_path, 'w') as f:
        for obj in root.findall('object'):
            class_name = obj.find('name').text.strip()
            if class_name not in class_mapping:
                print(f"⚠️ Unknown class '{class_name}' in {xml_path}")
                continue

            class_id = class_mapping[class_name]
            bbox = obj.find('bndbox')

            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # Normalize to [0,1]
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def get_image_dimensions(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    return width, height


def find_image_file(base_path, stem):
    """
    Find image file (jpg/png/etc.) corresponding to XML.
    """
    exts = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']
    for ext in exts:
        img_path = list(base_path.glob(f"**/{stem}{ext}"))
        if img_path:
            return img_path[0]
    return None


def auto_detect_classes(dataset_path, sample_size=50):
    """
    Auto-detect unique class names from XML annotations.
    """
    dataset_path = Path(dataset_path)
    class_names = set()
    for xml_file in dataset_path.glob('**/*.xml'):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name').text.strip()
                class_names.add(name)
                if len(class_names) > sample_size:
                    break
        except:
            continue
    class_names = sorted(list(class_names))
    mapping = {name: idx for idx, name in enumerate(class_names)}
    print(f"✅ Auto-detected classes: {class_names}")
    return mapping


def convert_dataset_xml_to_yolo(dataset_path, output_path, class_mapping=None):
    """
    Convert an entire dataset with XMLs into YOLO (v3/v5/v8) format.
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    splits = ['train', 'valid', 'test']

    if class_mapping is None:
        class_mapping = auto_detect_classes(dataset_path)

    print(f"\n📘 Using class mapping: {class_mapping}\n")

    for split in splits:
        split_path = dataset_path / split
        if not split_path.exists():
            print(f"⚠️ Split folder missing: {split_path}")
            continue

        out_img_dir = output_path / split / 'images'
        out_lbl_dir = output_path / split / 'labels'
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        xml_files = list(split_path.glob('**/*.xml'))
        print(f"Processing '{split}' - {len(xml_files)} XML files")

        for xml_file in xml_files:
            try:
                stem = xml_file.stem
                image_file = find_image_file(split_path, stem)
                if image_file is None:
                    print(f"⚠️ No image found for {xml_file}")
                    continue

                w, h = get_image_dimensions(xml_file)
                label_path = out_lbl_dir / f"{stem}.txt"

                convert_xml_to_yolo(xml_file, w, h, label_path, class_mapping)
                shutil.copy2(image_file, out_img_dir / image_file.name)

            except Exception as e:
                print(f"❌ Error: {xml_file} - {e}")

    create_data_yaml(output_path, list(class_mapping.keys()))
    print("\n✅ Conversion completed successfully!")
    print(f"YOLOv3 dataset ready at: {output_path}")
    return class_mapping


def create_data_yaml(output_path, class_names):
    """
    Create data.yaml for YOLOv3/v5/v8 training.
    """
    data = {
        'train': f'{output_path}/train/images',
        'val': f'{output_path}/valid/images',
        'test': f'{output_path}/test/images',
        'nc': len(class_names),
        'names': class_names
    }
    yaml_path = Path(output_path) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"🧾 Created data.yaml → {yaml_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert Pascal VOC XML dataset to YOLOv3 format")
    parser.add_argument('--dataset-path', type=str, default='./Dataset', help="Path to dataset with XMLs")
    parser.add_argument('--output-path', type=str, default='blood_cell_yolo', help="Output YOLO folder")
    parser.add_argument('--classes', nargs='+', help="Optional class list (in order)")

    args = parser.parse_args()

    if args.classes:
        mapping = {cls: i for i, cls in enumerate(args.classes)}
        print(f"🔖 Using provided class list: {args.classes}")
    else:
        mapping = None

    convert_dataset_xml_to_yolo(args.dataset_path, args.output_path, mapping)


if __name__ == "__main__":
    main()
