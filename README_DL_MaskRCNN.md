
# DL_MaskRCNN

This repository contains our Deep Learning final project focused on implementing and analyzing **Mask R-CNN** for object detection and instance segmentation. The work was completed as part of the coursework at **Oregon State University**.

## 📌 Project Summary

Mask R-CNN extends Faster R-CNN by adding a branch for predicting object masks at the pixel level. We implemented the full pipeline from training to evaluation using the COCO 2017 dataset. This project explores the effectiveness of different backbones, optimization strategies, and loss functions in real-world scenarios.

## 👩‍💻 Authors
- Aishwarya Joshi  
- Keerthana Gopalakrishnan  
- Sanika Prashant Deshmukh  

## 📁 Files Overview

| File | Description |
|------|-------------|
| `MASK_R_CNN_2.ipynb` & variations | Main Mask R-CNN implementation notebooks |
| `Object Detection.ipynb` | Base object detection before instance segmentation |
| `Object Detection and masking.ipynb` | Combining object detection and mask prediction |
| `ResNet101_50epoch_50k_Object_Detection.ipynb` | Mask R-CNN using ResNet101 on 50k images for 50 epochs |
| `ResNet101_20k_Object_Detection.ipynb` | Experiments with smaller dataset and ResNet101 |
| `Resnet50 -50epoch-50KObject Detection.ipynb` | ResNet50 baseline model training |
| `README.md` | Project summary and guidance |

## 📊 Methodology

- **Backbones used**: ResNet50 + FPN, ResNet101 + FPN
- **Datasets**: COCO 2017 (training on subsets of 20K, 50K, 80K)
- **Key Techniques**:
  - Region Proposal Networks (RPN)
  - RoIAlign for precise spatial feature alignment
  - Separate heads for classification, bounding box regression, and mask prediction
  - Focal Loss, IoU Loss, Dice Loss for improved training
- **Hardware Used**: NVIDIA A100 and V100 GPUs

## 📈 Results

- Achieved up to **47% validation accuracy** using ResNet101 + FPN trained for 100 epochs.
- ResNet101 consistently outperformed ResNet50.
- Mask R-CNN successfully generated precise instance masks and bounding boxes on COCO test images.
- Challenges included label ambiguity, overlapping object segmentation, and training time on large datasets.

## 🔍 Key Insights

- RoIAlign significantly boosts segmentation accuracy over traditional RoIPool.
- FPN improves detection across scales.
- Loss function changes (e.g., Dice, Focal) yield better performance on small datasets.
- Deep networks (ResNet101) benefit from longer training with lower learning rates.

## 📌 Future Work

- Real-time optimization (e.g., pruning, quantization)
- Lightweight architectures for edge deployment
- Incorporating depth/LiDAR data
- Better handling of occluded or overlapping instances

## 🧠 References

1. He et al., **Mask R-CNN**, ICCV 2017  
2. Girshick et al., **Faster R-CNN**, NIPS 2015  
3. Long et al., **FCN for Semantic Segmentation**, CVPR 2015  

---

## 📎 Report

Full project report with methodology and results is available [here](https://github.com/sanikadeshmukh/DL_MaskRCNN).

---

## 🧪 Usage

This project uses Python and PyTorch. Set up your environment and GPU to run the Jupyter notebooks:

```bash
pip install -r requirements.txt
jupyter notebook
```

---

## 📄 License

This repository is for academic and non-commercial research purposes only.
