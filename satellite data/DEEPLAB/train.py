from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from models.deeplabv3 import DeepLabv3  
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


root_path = os.path.join("/home/imana", "0_analysis", "data_select")
image_dir = os.path.join(root_path, "base_imgs")
mask_dir = os.path.join(root_path, "base_masks")


assert os.path.exists(image_dir), f"Image directory not found: {image_dir}"
assert os.path.exists(mask_dir), f"Mask directory not found: {mask_dir}"


image_paths = sorted([
    os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
    if os.path.isfile(os.path.join(image_dir, fname))
])
mask_paths = sorted([
    os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)
    if os.path.isfile(os.path.join(mask_dir, fname))
])


image_filenames = set(os.path.splitext(os.path.basename(path))[0] for path in image_paths)
mask_filenames = set(os.path.splitext(os.path.basename(path))[0].replace("_label", "") for path in mask_paths)
common_filenames = image_filenames.intersection(mask_filenames)

image_paths = [path for path in image_paths if os.path.splitext(os.path.basename(path))[0] in common_filenames]
mask_paths = [path for path in mask_paths if os.path.splitext(os.path.basename(path))[0].replace("_label", "") in common_filenames]

print(f"Number of images after filtering: {len(image_paths)}")
print(f"Number of masks after filtering: {len(mask_paths)}")

if len(image_paths) == 0 or len(mask_paths) == 0:
    raise ValueError("No matching images and masks found. Please check the dataset.")


train_idx, test_idx = train_test_split(range(len(image_paths)), test_size=0.2, random_state=42)


new_size = (256, 256)


class RoofDataset(Dataset):
    def __init__(self, indices, image_paths, mask_paths, new_size):
        self.indices = indices
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.new_size = new_size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        image = plt.imread(self.image_paths[i])
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        image = torch.tensor(image, dtype=torch.float32) / 255.0

        mask = plt.imread(self.mask_paths[i])
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=-1)
        mask = torch.tensor(mask, dtype=torch.long) 

     
        mask = torch.clamp(mask, min=0, max=1)  

       
        image = torch.nn.functional.interpolate(image.unsqueeze(0).permute(0, 3, 1, 2), size=self.new_size, mode='bilinear').squeeze(0)
        
       
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0).permute(0, 3, 1, 2).float(), size=self.new_size, mode='nearest').squeeze(0).long()

        return image, mask


train_dataset = RoofDataset(train_idx, image_paths, mask_paths, new_size)
test_dataset = RoofDataset(test_idx, image_paths, mask_paths, new_size)


train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


model = DeepLabv3(nc=2)  
model.to(device)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)


epochs = 50
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0
    correct_pixels = 0
    total_pixels = 0

    print(f"Epoch {epoch}/{epochs}")

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

       
        if outputs.shape[1] == 1:
            outputs = outputs.squeeze(1)  

        masks = masks.squeeze(1)  

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

       
        predictions = torch.argmax(outputs, dim=1)  
        correct_pixels += (predictions == masks).sum().item()
        total_pixels += masks.numel()

    train_losses.append(epoch_loss / len(train_loader))
    train_accuracies.append(correct_pixels / total_pixels)
    print(f"Epoch {epoch} Loss: {epoch_loss / len(train_loader):.4f}, Accuracy: {correct_pixels / total_pixels:.4f}")

    
    model.eval()
    val_loss = 0
    val_correct_pixels = 0
    val_total_pixels = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)

            if outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)

            masks = masks.squeeze(1)

            loss = criterion(outputs, masks)
            val_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            val_correct_pixels += (predictions == masks).sum().item()
            val_total_pixels += masks.numel()

    val_losses.append(val_loss / len(test_loader))
    val_accuracies.append(val_correct_pixels / val_total_pixels)
    print(f"Validation Loss: {val_losses[-1]:.4f}, Validation Accuracy: {val_accuracies[-1]:.4f}")

    
    checkpoint_path = os.path.join(os.getcwd(), f"deeplab_epoch_{epoch:02d}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved at {checkpoint_path}")


final_model_path = os.path.join(os.getcwd(), "deeplab_final.pth")
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved at {final_model_path}")


history_df = pd.DataFrame({
    "epoch": range(1, epochs + 1),
    "train_loss": train_losses,
    "train_accuracy": train_accuracies,
    "val_loss": val_losses,
    "val_accuracy": val_accuracies
})
history_df.to_csv(os.path.join(os.getcwd(), "training_scores.csv"), index=False)
print("Training scores saved as 'training_scores.csv'.")
