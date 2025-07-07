import os
import numpy as np
from skimage import io, exposure

def match_colors(source_image, target_image):
    
    if source_image.shape[-1] != target_image.shape[-1]:
        raise ValueError("source")

    
    matched_image = np.zeros_like(source_image)
    for channel in range(source_image.shape[-1]): 
        matched_image[..., channel] = exposure.match_histograms(
            source_image[..., channel], target_image[..., channel]
        )

    return matched_image


source_folder = "/home/imana/0_analysis/data_select/manual_imgs"  
target_image_path = "/home/imana/0_analysis/data_select/base_imgs/solarpanels_native_1__x0_0_y0_6963_dxdy_416.bmp" 
output_folder = "/home/imana/0_analysis/data_select/output/ADAPT/COLOR_MATCHING/cl_match"  


os.makedirs(output_folder, exist_ok=True)


target_image = io.imread(target_image_path)


if target_image.shape[-1] != 3:
    raise ValueError("The target image must be an RGB image.")


for filename in os.listdir(source_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        source_image_path = os.path.join(source_folder, filename)
        output_image_path = os.path.join(output_folder, filename)

        
        source_image = io.imread(source_image_path)

        
        if source_image.shape[-1] != 3:
            print(f"Skipping non-RGB image: {filename}")
            continue

       
        print(f"Matching colors for: {filename}")
        matched_image = match_colors(source_image, target_image)

       
        io.imsave(output_image_path, matched_image)
        print(f"Color-matched image saved to: {output_image_path}")

print("completed")
