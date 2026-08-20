import os
import json
import glob

# script is in backend/src/scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, "../../.."))

media_info_dir = os.path.join(root_dir, "frontend/public/media-info")
images_dir = os.path.join(root_dir, "backend/processed/images")

json_files = glob.glob(os.path.join(media_info_dir, "*.json"))
count = 0

for filepath in json_files:
    basename = os.path.basename(filepath)
    video_name = os.path.splitext(basename)[0] # e.g. L30_V014
    
    parts = video_name.split('_')
    if len(parts) != 2:
        continue
    
    k_folder = parts[0]
    v_id = parts[1]
    
    img_folder = os.path.join(images_dir, k_folder, v_id)
    
    if os.path.isdir(img_folder):
        frames = []
        for f in os.listdir(img_folder):
            if f.endswith('.jpg'):
                frame_idx = os.path.splitext(f)[0]
                frames.append(int(frame_idx))
        
        frames.sort()
        list_frames = [str(f) for f in frames]
        
        with open(filepath, 'r') as f:
            try:
                data = json.load(f)
            except:
                continue
                
        data['listFrames'] = list_frames
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        
        count += 1

print(f"Added listFrames to {count} JSON files based on extracted images!")
