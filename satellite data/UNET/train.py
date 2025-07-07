import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import segmentation_models as sm

print("GPUs available:", tf.config.list_physical_devices('GPU'))

os.environ["SM_FRAMEWORK"] = "tf.keras"
os.chdir('/home/imana/0_analysis/UNET')

BACKBONE = 'resnet18'
preprocess_input = sm.get_preprocessing(BACKBONE)

root_path = os.path.join("/home/imana", "0_analysis", "data_select")
image_dir = os.path.join(root_path, "base_imgs")
mask_dir = os.path.join(root_path, "base_masks")

# Output paths
output_root = "/home/imana/0_analysis/data_select/output/UNET"
csv_path = os.path.join(output_root, "csv", "training_scores.csv")
checkpoints_path = os.path.join(output_root, "checkpoints", "model_best.h5")
final_model_path = os.path.join(output_root, "model_final.h5")
weights_h5_path = os.path.join(output_root, "weights", "model_weights.h5")
weights_pth_path = os.path.join(output_root, "weights", "model_weights.pth")
extra_path = os.path.join(output_root, "extra", "example_kernel.png")

# Ensure directories exist
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
os.makedirs(os.path.dirname(checkpoints_path), exist_ok=True)
os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
os.makedirs(os.path.dirname(weights_h5_path), exist_ok=True)
os.makedirs(os.path.dirname(extra_path), exist_ok=True)

# Collect image and mask paths, excluding directories
image_paths = sorted([
    os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
    if os.path.isfile(os.path.join(image_dir, fname))
])
mask_paths = sorted([
    os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)
    if os.path.isfile(os.path.join(mask_dir, fname))
])

# Match images and masks by filename
image_filenames = set(os.path.splitext(os.path.basename(path))[0] for path in image_paths)
mask_filenames = set(os.path.splitext(os.path.basename(path))[0].replace("_label", "") for path in mask_paths)

common_filenames = image_filenames.intersection(mask_filenames)

image_paths = [path for path in image_paths if os.path.splitext(os.path.basename(path))[0] in common_filenames]
mask_paths = [path for path in mask_paths if os.path.splitext(os.path.basename(path))[0].replace("_label", "") in common_filenames]

print(f"Number of images after filtering: {len(image_paths)}")
print(f"Number of masks after filtering: {len(mask_paths)}")

if len(image_paths) == 0 or len(mask_paths) == 0:
    raise ValueError("No matching images and masks found. Please check the dataset.")

# Split into training and testing sets
train_idx, test_idx = train_test_split(range(len(image_paths)), test_size=0.2, random_state=42)

# Resize all images and masks to this size
new_size = (256, 256)

# Dataset generators
def train_dataset_generator():
    for i in train_idx:
        try:
            # Load and preprocess image
            image = plt.imread(image_paths[i])
            if image.ndim == 2:  # Grayscale to RGB
                image = np.stack((image,) * 3, axis=-1)
            image = tf.convert_to_tensor(image, dtype=tf.float32) / 255.0
            image = tf.image.resize(image, new_size)

            # Load and preprocess mask
            mask = plt.imread(mask_paths[i])
            if mask.ndim == 2:  # Grayscale to single channel
                mask = np.expand_dims(mask, axis=-1)
            mask = tf.convert_to_tensor(mask, dtype=tf.float32) / 255.0
            mask = tf.image.resize(mask, new_size)

            # Set shapes
            image.set_shape((*new_size, 3))
            mask.set_shape((*new_size, 1))

            yield image, mask
        except Exception as e:
            print(f"Error processing file: {image_paths[i]} or {mask_paths[i]} - {e}")

def test_dataset_generator():
    for i in test_idx:
        try:
            # Load and preprocess image
            image = plt.imread(image_paths[i])
            if image.ndim == 2:  # Grayscale to RGB
                image = np.stack((image,) * 3, axis=-1)
            image = tf.convert_to_tensor(image, dtype=tf.float32) / 255.0
            image = tf.image.resize(image, new_size)

            # Load and preprocess mask
            mask = plt.imread(mask_paths[i])
            if mask.ndim == 2:  # Grayscale to single channel
                mask = np.expand_dims(mask, axis=-1)
            mask = tf.convert_to_tensor(mask, dtype=tf.float32) / 255.0
            mask = tf.image.resize(mask, new_size)

            # Set shapes
            image.set_shape((*new_size, 3))
            mask.set_shape((*new_size, 1))

            yield image, mask
        except Exception as e:
            print(f"Error processing file: {image_paths[i]} or {mask_paths[i]} - {e}")

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_generator(
    train_dataset_generator,
    output_signature=(
        tf.TensorSpec(shape=(*new_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(*new_size, 1), dtype=tf.float32),
    ),
).batch(8).repeat()

test_dataset = tf.data.Dataset.from_generator(
    test_dataset_generator,
    output_signature=(
        tf.TensorSpec(shape=(*new_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(*new_size, 1), dtype=tf.float32),
    ),
).batch(8).repeat()

# Verify dataset shapes
for image, mask in train_dataset.take(1):
    print("Image shape in dataset:", image.shape)
    print("Mask shape in dataset:", mask.shape)

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"GPUs available: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found.")

# Build and compile the model
model = sm.Unet(BACKBONE, encoder_weights='imagenet', input_shape=(256, 256, 3), classes=1, activation='sigmoid')

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Training parameters
steps_per_epoch = len(train_idx) // 8
validation_steps = len(test_idx) // 8

# Train the model
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=50,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(checkpoints_path, save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ]
)

# Save the final model
try:
    model.save(final_model_path)
    print(f"Final model saved at: {final_model_path}")
except Exception as e:
    print(f"Error saving final model: {e}")

# Save model weights
try:
    model.save_weights(weights_h5_path)
    print(f"Model weights saved as .h5 at: {weights_h5_path}")
except Exception as e:
    print(f"Error saving weights as .h5: {e}")

try:
    model.save_weights(weights_pth_path)
    print(f"Model weights saved as .pth at: {weights_pth_path}")
except Exception as e:
    print(f"Error saving weights as .pth: {e}")

# Save the training scores
try:
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(csv_path, index=False)
    print(f"Training scores saved at: {csv_path}")
except Exception as e:
    print(f"Error saving training scores: {e}")

# Save an example kernel visualization
try:
    kernel = model.get_layer(index=1).get_weights()[0][:, :, :, 0]  # Example kernel
    plt.imshow(kernel[:, :, 0], cmap='viridis')
    plt.colorbar()
    plt.title("Example Kernel")
    plt.savefig(extra_path)
    print(f"Example kernel visualization saved at: {extra_path}")
except Exception as e:
    print(f"Error saving example kernel visualization: {e}")
