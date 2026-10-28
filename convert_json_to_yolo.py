import json
import os
import glob

def convert_labelme_to_yolo(input_dir=None, output_dir=None):
    if input_dir is None:
        input_dir = os.path.join(os.path.dirname(__file__), 'manual_dataset/labels/train/train_json')
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'manual_dataset/labels/train')
    
    os.makedirs(output_dir, exist_ok=True)
    
    json_files = glob.glob(os.path.join(input_dir, '*.json'))
    print(f"🔄 Found {len(json_files)} LabelMe JSON files. Converting to YOLO format...")
    
    converted_count = 0
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
            
        img_w = data['imageWidth']
        img_h = data['imageHeight']
        
        base_name = os.path.splitext(os.path.basename(jf))[0]
        out_txt = os.path.join(output_dir, f"{base_name}.txt")
        
        yolo_lines = []
        for shape in data['shapes']:
            points = shape['points']
            if not points: continue
            
            # 从多边形点集中提取最小外接矩形 (Bounding Box)
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            xmin, xmax = min(x_coords), max(x_coords)
            ymin, ymax = min(y_coords), max(y_coords)
            
            # 计算 YOLO 需要的中心点和宽高
            x_center = (xmin + xmax) / 2.0
            y_center = (ymin + ymax) / 2.0
            width = xmax - xmin
            height = ymax - ymin
            
            # 归一化 (除以图片实际宽和高)
            norm_x = x_center / img_w
            norm_y = y_center / img_h
            norm_w = width / img_w
            norm_h = height / img_h
            
            # 所有类别强制映射为 0 (代表 fish)
            yolo_lines.append(f"0 {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}")
            
        with open(out_txt, 'w') as f:
            f.write("\n".join(yolo_lines))
            
        converted_count += 1
            
    print(f"✅ Conversion complete! {converted_count} .txt files created in {output_dir}")

if __name__ == "__main__":
    convert_labelme_to_yolo()
