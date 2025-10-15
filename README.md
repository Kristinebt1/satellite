Robust PV segmentation across new domains (different countries, lighting conditions, and sensors) without retraining.
This project focuses on improving the generalization of segmentation models for rooftop photovoltaic (PV) panels using inference-time domain adaptation.
Models such as U-Net, SegNet, and DeepLabv3 were trained on a controlled MAXAR dataset from Southern Germany and tested on Norwegian aerial imagery.
The study explores techniques like CLAHE, Reinhard Color Normalization (CLN), Histogram Matching, and DANN to enhance cross-domain performance.

Findings:

Baseline (in-domain): ~99.2% (U-Net), 99.4% (SegNet), 98.4% (DeepLabv3).

Out-of-domain: High pixel accuracy may hide low IoU/F1 — U-Net showed the most stable performance.

Best inference-time methods: CLAHE (+1.4pp accuracy), CLN improved detection of dark pc-Si panels in shadow, DANN helped reduce overfitting (especially for DeepLabv3), while Histogram Matching added noise.
