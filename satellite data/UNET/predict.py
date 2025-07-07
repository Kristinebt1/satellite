import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, confusion_matrix
from medpy.metric.binary import hd, dc, assd
import pandas as pd


root_path = os.path.join("/home/imana", "0_analysis", "data_select")
eval_image_dir = os.path.join(root_path, "eval_imgs")
eval_mask_dir = os.path.join(root_path, "eval_masks")
output_root = os.path.join(root_path, "output", "UNET")
csv_path = os.path.join(output_root, "csv", "prediction_metrics.csv")
jaccard_vis_path = os.path.join(output_root, "jaccard")
prediction_output_dir = os.path.join(output_root, "prediction")
overall_metrics_path = os.path.join(os.getcwd(), "overall_metrics.txt")  # Save in the current folder


os.makedirs(os.path.dirname(csv_path), exist_ok=True)
os.makedirs(jaccard_vis_path, exist_ok=True)
os.makedirs(prediction_output_dir, exist_ok=True)


model_path = os.path.join(output_root, "model_final.h5")
model = tf.keras.models.load_model(model_path, compile=False)
print(f"Model loaded from: {model_path}")


new_size = (256, 256)


eval_image_paths = sorted([
    os.path.join(eval_image_dir, fname) for fname in os.listdir(eval_image_dir)
    if os.path.isfile(os.path.join(eval_image_dir, fname))
])
eval_mask_paths = sorted([
    os.path.join(eval_mask_dir, fname) for fname in os.listdir(eval_mask_dir)
    if os.path.isfile(os.path.join(eval_mask_dir, fname))
])


num_images_to_test = 200  # Replace this with the desired number of images
eval_image_paths = eval_image_paths[:num_images_to_test]
eval_mask_paths = eval_mask_paths[:num_images_to_test]


metrics = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1_score": [],
    "specificity": [],
    "dice_similarity": [],
    "iou": [],
    "hausdorff_distance": [],
    "absolute_volume_difference": [],
    "jaccard_index": []
}


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true.flatten(), y_pred.flatten()).ravel()
    return tn / (tn + fp)


for i, (image_path, mask_path) in enumerate(zip(eval_image_paths, eval_mask_paths)):
    # Load and preprocess image
    image = plt.imread(image_path)
    if image.ndim == 2:  # Grayscale to RGB
        image = np.stack((image,) * 3, axis=-1)
    image = tf.image.resize(image, new_size) / 255.0
    image = tf.expand_dims(image, axis=0)  # Add batch dimension

   
    mask = plt.imread(mask_path)
    if mask.ndim == 2:  # Grayscale to single channel
        mask = np.expand_dims(mask, axis=-1)
    mask = tf.image.resize(mask, new_size, method='nearest').numpy()
    mask = (mask > 0.5).astype(np.uint8)  # Binarize mask

 
    prediction = model.predict(image)[0]
    prediction = (prediction > 0.5).astype(np.uint8)  # Binarize prediction

  
    prediction_save_path = os.path.join(prediction_output_dir, f"prediction_{i + 1}.png")
    plt.imsave(prediction_save_path, prediction[:, :, 0], cmap='gray')
    print(f"Saved prediction to: {prediction_save_path}")

    
    y_true = mask.flatten()
    y_pred = prediction.flatten()

 
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    specificity = specificity_score(y_true, y_pred)
    dice = dc(mask, prediction)
    iou = jaccard_score(y_true, y_pred, zero_division=1)

    # Check for empty masks before calculating Hausdorff Distance
    if np.sum(mask) == 0 or np.sum(prediction) == 0:
        hausdorff = np.nan  # Assign NaN if either mask or prediction is empty
    else:
        hausdorff = hd(mask, prediction)

    avd = np.abs(np.sum(mask.astype(np.float64)) - np.sum(prediction.astype(np.float64))) / np.sum(mask.astype(np.float64))
    jaccard = iou

   
    metrics["accuracy"].append(accuracy)
    metrics["precision"].append(precision)
    metrics["recall"].append(recall)
    metrics["f1_score"].append(f1)
    metrics["specificity"].append(specificity)
    metrics["dice_similarity"].append(dice)
    metrics["iou"].append(iou)
    metrics["hausdorff_distance"].append(hausdorff)
    metrics["absolute_volume_difference"].append(avd)
    metrics["jaccard_index"].append(jaccard)

    # Save Jaccard visualization
    jaccard_vis_save_path = os.path.join(jaccard_vis_path, f"jaccard_{i + 1}.png")
    plt.figure()
    plt.imshow(mask[:, :, 0], cmap='gray', alpha=0.5, label="Ground Truth")
    plt.imshow(prediction[:, :, 0], cmap='jet', alpha=0.5, label="Prediction")
    plt.title(f"Jaccard Index: {jaccard:.4f}")
    plt.colorbar()
    plt.savefig(jaccard_vis_save_path)
    plt.close()
    print(f"Saved Jaccard visualization to: {jaccard_vis_save_path}")


metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(csv_path, index=False)
print(f"Metrics saved to: {csv_path}")

overall_metrics = {
    "Overall Accuracy": np.mean(metrics["accuracy"]),
    "Overall Precision": np.mean(metrics["precision"]),
    "Overall Recall": np.mean(metrics["recall"]),
    "Overall F1-Score": np.mean(metrics["f1_score"]),
    "Overall Specificity": np.mean(metrics["specificity"]),
    "Overall Dice Similarity": np.mean(metrics["dice_similarity"]),
    "Overall IoU": np.mean(metrics["iou"]),
    "Overall Hausdorff Distance": np.mean(metrics["hausdorff_distance"]),
    "Overall Absolute Volume Difference": np.mean(metrics["absolute_volume_difference"]),
    "Overall Jaccard Index": np.mean(metrics["jaccard_index"])
}

with open(overall_metrics_path, "w") as f:
    for metric, value in overall_metrics.items():
        f.write(f"{metric}: {value:.4f}\n")
print(f"Overall metrics saved to: {overall_metrics_path}")
