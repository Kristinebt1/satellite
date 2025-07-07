import os
import torch
import numpy as np
from PIL import Image
from skimage.io import imread
from skimage.transform import resize
from models.deeplabv3 import DeepLabv3  


input_folder = "input" 
output_folder = "output" 
model_path = os.path.join("checkpoints", "deeplab_final.pth") 


os.makedirs(output_folder, exist_ok=True)


print("Loading model.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepLabv3(nc=2)  
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully.")


model_input_size = (256, 256)


print("Processing images...")
for filename in os.listdir(input_folder):
    if filename.endswith(".bmp"):  
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".bmp", ".png"))

        
        image = imread(input_path)
        original_size = image.shape[:2]
        image_resized = resize(image, model_input_size, anti_aliasing=True, preserve_range=True)
        image_resized = torch.tensor(image_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        image_resized = image_resized.to(device)

        
        with torch.no_grad():
            output = model(image_resized)
            predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        
        predicted_mask_resized = resize(predicted_mask, original_size, order=0, preserve_range=True, anti_aliasing=False)

        
        raw_mask_path = os.path.join(output_folder, filename.replace(".bmp", "_raw.png"))
        raw_predicted_mask = (predicted_mask_resized * 255).astype(np.uint8)
        Image.fromarray(raw_predicted_mask, mode="L").save(raw_mask_path)
        print(f"Saved raw predicted mask to: {raw_mask_path}")

        
        threshold = 0.5
        predicted_mask_binary = (predicted_mask_resized > threshold).astype(np.uint8) * 255

    
        binary_mask_path = os.path.join(output_folder, filename.replace(".bmp", "_binary.png"))
        predicted_mask_image = Image.fromarray(predicted_mask_binary, mode="L")
        predicted_mask_image.save(binary_mask_path)
        print(f"Saved binary predicted mask to: {binary_mask_path}")

print("Prediction completed.")
