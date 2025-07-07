import os
import numpy as np
import cv2
from skimage import io

def reinhard_normalization(source_image, target_image):
   
   
    source_lab = cv2.cvtColor(source_image, cv2.COLOR_RGB2LAB)
    target_lab = cv2.cvtColor(target_image, cv2.COLOR_RGB2LAB)

    
    source_mean, source_std = cv2.meanStdDev(source_lab)
    target_mean, target_std = cv2.meanStdDev(target_lab)

    
    source_mean = source_mean.reshape((1, 1, 3))
    source_std = source_std.reshape((1, 1, 3))
    target_mean = target_mean.reshape((1, 1, 3))
    target_std = target_std.reshape((1, 1, 3))

   
    normalized_lab = (source_lab - source_mean) / source_std * target_std + target_mean
    normalized_lab = np.clip(normalized_lab, 0, 255).astype(np.uint8)

    
    normalized_rgb = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2RGB)

    return normalized_rgb


source_folder = "/home/imana/0_analysis/data_select/manual_imgs"
average_target_image_path = "/home/imana/0_analysis/CLN/average_target_image.bmp"
output_folder = "/home/imana/0_analysis/data_select/cln_imgs"


target_image = io.imread(average_target_image_path)


os.makedirs(output_folder, exist_ok=True)


for filename in os.listdir(source_folder):
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".bmp"):
        source_image_path = os.path.join(source_folder, filename)
        output_image_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_cln.bmp")

        
        source_image = io.imread(source_image_path)

        
        normalized_image = reinhard_normalization(source_image, target_image)

      
        io.imsave(output_image_path, normalized_image)
        print(f"Normalized image saved to: {output_image_path}")
