# Fruit Classification Challenge

## Introduction
This project explores the development of an efficient deep learning model for classifying fruits into different categories. Leveraging state-of-the-art neural network architectures, including MobileNetV3, the project achieved a highly accurate and computationally efficient fruit classification pipeline. The model is optimized for real-world applications such as agriculture, food processing, and retail.

---

## Features
- **High Accuracy**: Achieved 99.59% test accuracy using MobileNetV3.
- **Lightweight Architecture**: Optimized for fast inference and resource-constrained environments.
- **Data Augmentation**: Utilized techniques like random rotations, color jittering, and horizontal flips to improve model robustness.
- **Transfer Learning**: Pre-trained models on ImageNet for efficient feature extraction.
- **Real-World Applicability**: Suitable for tasks like automated sorting and fruit recognition in stores.

---

## Results
The best-performing model, **MobileNetV3**, achieved:
- **Test Accuracy**: 99.59%
- **Average Inference Time**: 0.0068 seconds per image

Other tested architectures include Vision Transformer (ViT), ResNet50, EfficientNet, and DenseNet, each evaluated for accuracy and computational efficiency.

---

## Demo
A video demonstration of the project is available in the `Demo` folder:
- **Path**: Demo/Demo Presentation.mp4

---

## Documentation
The full project report, detailing the methodology, experiments, and results, can be found in the `Documentation` folder:
- **Path**: Documentation/Final Report.pdf

---

## Contribution
The project was collaboratively developed by **Group 14**:
- **Zeeshan Ansari (510370813)**
- **Syed Hamza Kaliyadan (500585454)**

Each member contributed equally to model development, experimentation, and report preparation.

---

## Prerequisites
To run this project, you need:
1. **Python 3.8+**
2. **Required Libraries**:
   - PyTorch
   - Torchvision
   - NumPy
   - Matplotlib
   - Scikit-learn
3. **Hardware**:
   - A CUDA-enabled GPU for faster training (optional but recommended)

Install the dependencies using:
```bash 
python -m pip install -r requirements.txt
```

---

## Usage
### Training
To train the model, run the following script:
```python 
[python project2_train.py]
```

### Testing
To evaluate the model's performance:
```python 
[python project2_test.py]
```

Ensure that the trained model weights (project2.pth) are in the same directory as the scripts.

---
