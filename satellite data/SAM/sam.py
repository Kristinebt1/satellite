import os
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm  


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


input_folder = "imgs"
output_folder = "masks"
labels_folder = "labels"
checkpoints_folder = "checkpoints"

if not os.path.exists(input_folder):
    raise ValueError(f"Input folder '{input_folder}' does not exist.")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(checkpoints_folder, exist_ok=True)


checkpoint_path = os.path.join(checkpoints_folder, "sam_vit_h_4b8939.pth")
if not os.path.exists(checkpoint_path):
    raise ValueError(f"Checkpoint file '{checkpoint_path}' not found.")
print(f"Using checkpoint: {checkpoint_path}")


sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
state_dict = torch.load(checkpoint_path, map_location=device)
sam.load_state_dict(state_dict)
sam.to(device=device)
predictor = SamPredictor(sam)


def process_image_with_labels(img_path, label_path):
    
    img = cv2.imread(img_path)
    height, width, _ = img.shape

    
    with open(label_path, "r") as f:
        lines = f.readlines()

    input_boxes = []

    for line in lines:
        parts = line.strip().split()
        class_id, x_center, y_center, box_width, box_height = map(float, parts)

       
        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height

        
        x_min = int(x_center - box_width / 2)
        y_min = int(y_center - box_height / 2)
        x_max = int(x_center + box_width / 2)
        y_max = int(y_center + box_height / 2)

        
        input_boxes.append([x_min, y_min, x_max, y_max])

    
    predictor.set_image(img)

    
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    
    for box in input_boxes:
        box = np.array(box).reshape(1, -1)  
        masks, _, _ = predictor.predict(
            box=box,
            multimask_output=False  
        )

       
        mask = masks[0]  
        combined_mask = np.maximum(combined_mask, mask)

   
    binary_mask = (combined_mask > 0).astype(np.uint8) * 255

   
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)  
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)  

    
    save_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(img_path))[0]}_mask.bmp")
    cv2.imwrite(save_path, binary_mask)
    print(f"Saved binary mask to {save_path}")


def process_all_images():
   
    
    image_files = [f for f in sorted(os.listdir(input_folder)) if f.lower().endswith(".tif")]
    if not image_files:
        print("No .tif")
        return

    
    for img_file in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(input_folder, img_file)
        label_file = f"{os.path.splitext(img_file)[0]}.txt"
        label_path = os.path.join(labels_folder, label_file)

        if not os.path.exists(label_path):
            print(f"Label file not found for {img_file}. Skipping...")
            continue

        process_image_with_labels(img_path, label_path)

    print("All images processed.")


process_all_images()
