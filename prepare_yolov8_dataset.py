import xml.etree.ElementTree as ET
import os
import glob
from pathlib import Path
import yaml

def convert_xml_to_yolo(xml_path, img_width, img_height, output_path):
    """
    Convert XML annotation to YOLO format
    
    Args:
        xml_path: Path to XML file
        img_width: Image width
        img_height: Image height
        output_path: Output text file path
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    with open(output_path, 'w') as f:
        for obj in root.findall('object'):
            # Get class name
            class_name = obj.find('name').text
            
            # Get bounding box
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Convert to YOLO format (normalized center coordinates)
            x_center = (xmin + xmax) / 2.0 / img_width
            y_center = (ymin + ymax) / 2.0 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            # Write to file (class_id x_center y_center width height)
            # We'll map class names to IDs later
            f.write(f"{class_name} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def get_image_dimensions(xml_path):
    """
    Extract image dimensions from XML file
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    return width, height

def convert_dataset_xml_to_yolo(dataset_path, output_path, class_mapping=None):
    """
    Convert entire dataset from XML to YOLO format
    
    Args:
        dataset_path: Path to original dataset
        output_path: Path for YOLO formatted dataset
        class_mapping: Dictionary mapping class names to IDs
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    splits = ['train', 'valid', 'test']
    
    # Auto-detect class names if not provided
    if class_mapping is None:
        class_mapping = auto_detect_classes(dataset_path)
    
    print(f"Class mapping: {class_mapping}")
    
    # Process each split
    for split in splits:
        split_path = dataset_path / split
        if not split_path.exists():
            print(f"Warning: {split_path} does not exist, skipping...")
            continue
        
        # Create output directories
        images_output_dir = output_path / split / 'images'
        labels_output_dir = output_path / split / 'labels'
        images_output_dir.mkdir(parents=True, exist_ok=True)
        labels_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find XML files
        xml_files = list(split_path.glob('**/*.xml'))
        
        # Also check in annotations subfolder
        annotations_dir = split_path / 'annotations'
        if annotations_dir.exists():
            xml_files.extend(annotations_dir.glob('*.xml'))
        
        print(f"Processing {split}: {len(xml_files)} XML files found")
        
        converted_count = 0
        for xml_file in xml_files:
            try:
                # Get corresponding image file
                image_stem = xml_file.stem
                image_file = find_image_file(split_path, image_stem)
                
                if image_file is None:
                    print(f"Warning: No image found for {xml_file}")
                    continue
                
                # Get image dimensions from XML
                img_width, img_height = get_image_dimensions(xml_file)
                
                # Create output label path
                label_output_path = labels_output_dir / f"{image_stem}.txt"
                
                # Convert XML to YOLO format
                convert_xml_to_yolo_with_mapping(xml_file, img_width, img_height, 
                                               label_output_path, class_mapping)
                
                # Copy image to output directory
                import shutil
                shutil.copy2(image_file, images_output_dir / image_file.name)
                
                converted_count += 1
                
            except Exception as e:
                print(f"Error processing {xml_file}: {e}")
        
        print(f"Converted {converted_count} files in {split}")
    
    # Create data.yaml
    create_data_yaml(output_path, list(class_mapping.keys()))
    
    return class_mapping

def convert_xml_to_yolo_with_mapping(xml_path, img_width, img_height, output_path, class_mapping):
    """
    Convert XML to YOLO format using class mapping
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    with open(output_path, 'w') as f:
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            
            # Map class name to ID
            if class_name not in class_mapping:
                print(f"Warning: Unknown class '{class_name}' in {xml_path}")
                continue
            
            class_id = class_mapping[class_name]
            
            # Get bounding box
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Convert to YOLO format
            x_center = (xmin + xmax) / 2.0 / img_width
            y_center = (ymin + ymax) / 2.0 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            # Ensure coordinates are within [0,1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def find_image_file(base_path, stem):
    """
    Find image file with given stem in various formats
    """
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
        # Check in images folder
        images_dir = base_path / 'images'
        if images_dir.exists():
            image_file = images_dir / f"{stem}{ext}"
            if image_file.exists():
                return image_file
        
        # Check in base directory
        image_file = base_path / f"{stem}{ext}"
        if image_file.exists():
            return image_file
    
    return None

def auto_detect_classes(dataset_path, sample_size=50):
    """
    Auto-detect class names from XML files
    """
    dataset_path = Path(dataset_path)
    class_names = set()
    
    splits = ['train', 'vaild', 'test']
    
    for split in splits:
        split_path = dataset_path / split
        if not split_path.exists():
            continue
        
        # Find XML files
        xml_files = list(split_path.glob('**/*.xml'))
        annotations_dir = split_path / 'annotations'
        if annotations_dir.exists():
            xml_files.extend(annotations_dir.glob('*.xml'))
        
        # Sample some files to detect classes
        for xml_file in xml_files[:sample_size]:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    class_names.add(class_name)
            except:
                continue
    
    # Create mapping
    class_names = sorted(list(class_names))
    class_mapping = {name: idx for idx, name in enumerate(class_names)}
    
    print(f"Auto-detected classes: {class_names}")
    return class_mapping

def create_data_yaml(dataset_path, class_names):
    """
    Create data.yaml file for YOLOv8
    """
    dataset_path = Path(dataset_path)
    
    data = {
        'path': str(dataset_path.absolute()),
        'train': 'train/Images',
        'val': 'vaild/Images',
        'test': 'test/Images',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = dataset_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"Created data.yaml at: {yaml_path}")
    return yaml_path

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert XML annotations to YOLO format')
    parser.add_argument('--dataset-path', type=str, default="./Dataset", help='Path to original dataset with XML annotations')
    parser.add_argument('--output-path', type=str, default='blood_cell_yolo', help='Output path for YOLO formatted dataset')
    parser.add_argument('--classes', type=str, nargs='+', help='Class names in order (optional)')
    
    args = parser.parse_args()
    
    # Prepare class mapping
    if args.classes:
        class_mapping = {name: idx for idx, name in enumerate(args.classes)}
        print(f"Using specified classes: {args.classes}")
    else:
        class_mapping = None
    
    # Convert dataset
    convert_dataset_xml_to_yolo(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        class_mapping=class_mapping
    )
    
    print(f"\n✅ Conversion completed! YOLO dataset saved to: {args.output_path}")
    print(f"📁 Dataset structure:")
    print(f"   - {args.output_path}/train/images/")
    print(f"   - {args.output_path}/train/labels/")
    print(f"   - {args.output_path}/valid/images/")
    print(f"   - {args.output_path}/valid/labels/")
    print(f"   - {args.output_path}/test/images/")
    print(f"   - {args.output_path}/data.yaml")

if __name__ == "__main__":
    main()