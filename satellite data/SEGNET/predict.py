import os
import argparse
import numpy as np
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
from model import segnet_4_encoder_decoder 
import pandas as pd 


print("GPUs available:", tf.config.list_physical_devices('GPU'))


os.environ["SM_FRAMEWORK"] = "tf.keras"


INPUT_SHAPE = (256, 256, 3) 
N_LABELS = 1  
OUTPUT_MODE = "sigmoid"  
BATCH_SIZE = 8
EPOCHS = 50


root_path = os.path.join("/home/imana", "0_analysis", "data_select")
image_dir = os.path.join(root_path, "base_imgs")
mask_dir = os.path.join(root_path, "base_masks")

output_root = os.path.join(root_path, "output", "SEGNET")
csv_path = os.path.join(output_root, "csv", "training_scores.csv")
checkpoints_path = os.path.join(output_root, "checkpoints")
weights_path = os.path.join(output_root, "weights")  
final_model_path = os.path.join(output_root, "segnet_model_final.keras")
final_weights_path = os.path.join(weights_path, "segnet_final_weights.h5")  


os.makedirs(os.path.dirname(csv_path), exist_ok=True)
os.makedirs(checkpoints_path, exist_ok=True)
os.makedirs(weights_path, exist_ok=True)


parser = argparse.ArgumentParser(description="Train SegNet Model")
parser.add_argument("--num_images", type=int, default=None, help="Number of images to use for training and validation")
args = parser.parse_args()


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


if args.num_images:
    image_paths = image_paths[:args.num_images]
    mask_paths = mask_paths[:args.num_images]

print(f"Number of images after filtering: {len(image_paths)}")
print(f"Number of masks after filtering: {len(mask_paths)}")

if len(image_paths) == 0 or len(mask_paths) == 0:
    raise ValueError("No matching images and masks found. Please check the dataset.")


train_idx, val_idx = train_test_split(range(len(image_paths)), test_size=0.2, random_state=42)


new_size = (256, 256)


def iou(y_true, y_pred):
    
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)  # Threshold predictions to binary
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return tf.where(union > 0, intersection / union, 0.0)  # Avoid division by zero


def preprocess_image(image_path):
    "
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, new_size)
    return image / 255.0  # Normalize to [0, 1]

def preprocess_mask(mask_path):
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
    return np.expand_dims(mask, axis=-1) / 255.0  # Normalize to [0, 1] and add channel dimension

def dataset_generator(indices):
    
    for i in indices:
        yield preprocess_image(image_paths[i]), preprocess_mask(mask_paths[i])


train_dataset = tf.data.Dataset.from_generator(
    lambda: dataset_generator(train_idx),
    output_signature=(
        tf.TensorSpec(shape=(*new_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(*new_size, 1), dtype=tf.float32),
    ),
).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: dataset_generator(val_idx),
    output_signature=(
        tf.TensorSpec(shape=(*new_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(*new_size, 1), dtype=tf.float32),
    ),
).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)


for image, mask in train_dataset.take(1):
    print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")


model = segnet_4_encoder_decoder(
    input_shape=INPUT_SHAPE,
    batch_size=BATCH_SIZE,
    n_labels=N_LABELS,
    output_mode=OUTPUT_MODE,
    model_summary=True
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', iou]
)


steps_per_epoch = int(np.ceil(len(train_idx) / BATCH_SIZE))
validation_steps = int(np.ceil(len(val_idx) / BATCH_SIZE))


history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(weights_path, 'segnet_weights_epoch_{epoch:02d}.keras'),
            save_weights_only=True,
            save_best_only=False,
            monitor='val_loss',
            mode='min'
        ),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ]
)


model.save(final_model_path)
print(f"Final model saved at: {final_model_path}")


model.save_weights(final_weights_path)
print(f"Final model weights saved at: {final_weights_path}")


history_df = pd.DataFrame(history.history)
history_df.to_csv(csv_path, index=False)
print(f"Training scores saved at: {csv_path}")
