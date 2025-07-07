import os
import cv2


input_dir = "/home/imana/0_analysis/data_select/manual_imgs"
output_dir = "/home/imana/0_analysis/data_select/output/ADAPT/CLAHE/clahe_imgs"

os.makedirs(output_dir, exist_ok=True)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

for filename in os.listdir(input_dir):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
       
        img = cv2.imread(input_path)

        if img is None:
            print(f"Skipping invalid image: {input_path}")
            continue

        
        if len(img.shape) == 2: 
            clahe_img = clahe.apply(img)
        else:  
           
            b, g, r = cv2.split(img)

           
            b_clahe = clahe.apply(b)
            g_clahe = clahe.apply(g)
            r_clahe = clahe.apply(r)

            
            clahe_img = cv2.merge((b_clahe, g_clahe, r_clahe))

        
        cv2.imwrite(output_path, clahe_img)
        print(f"Processed and saved: {output_path}")
    else:
        print(f"Skipping non-image file: {input_path}")
