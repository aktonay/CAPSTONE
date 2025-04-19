# 🧠 Laryngeal Cancer Detection and Classification Using Deep Learning on Histopathological Images

Laryngeal cancer presents complex and often subtle symptoms, making early detection a significant challenge in medical diagnostics. This project leverages the power of deep learning and computer vision to classify histopathological tissue images into different stages of laryngeal cancer.

## 📌 Overview

Our research focuses on developing a robust and high-performing image classification model to aid in the early detection and classification of laryngeal cancer. Using deep learning architectures, we trained and tested multiple models on a curated and augmented dataset of histopathological images.

## 🧬 Dataset

- **Source**: Provided by an Italian researcher.
- **Total Images**: 1,320 original histopathological images.
- **After Augmentation**: 5,280 images.
- **Classes**:
  - **He**: Healthy
  - **Hbv**: Hypertrophic Blood Vessels
  - **IPCL**: IPCL-like Vessel
  - **Le**: Leukoplakia

The dataset was preprocessed and augmented (using techniques such as rotation, flipping, and contrast enhancement) to improve generalization and prevent overfitting.

## 🧠 Models Evaluated

We tested multiple deep-learning-based image classification models, including:

- ResNet50 ✅ (Best Performing)
- DenseNet121
- DenseNet201
- VGG16
- VGG19
- MobileNetV2
- InceptionV3

Each model was fine-tuned using transfer learning and evaluated on accuracy, precision, recall, and F1-score metrics.

## 🏆 Best Model

- **Model**: Fine-tuned ResNet50  
- **Accuracy**: 99.62%  
- This model outperformed all others and surpassed previous benchmark results in similar studies.  
- Emphasis was also placed on **precision** and **recall**, which are critical in disease classification tasks.

## 📊 Evaluation Metrics

We focused on multiple performance indicators:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

These metrics ensure the reliability and sensitivity of the model, especially for early-stage disease detection where false negatives must be minimized.

## 🔧 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy / Pandas
- Scikit-learn
- Matplotlib / Seaborn

## 📂 Project Structure

```
├── data/
│   ├── raw/                  # Original images
│   └── augmented/            # Augmented dataset
├── models/                   # Saved model weights
├── notebooks/                # Jupyter notebooks for training & evaluation
├── utils/                    # Helper functions (e.g., preprocessing, augmentation)
├── results/                  # Plots and evaluation reports
└── main.py                   # Training pipeline
```

## 🚀 Future Work

- Deploy the model in a clinical decision support tool.
- Incorporate segmentation to localize cancerous regions.
- Expand the dataset with multi-source histopathological images.

## 🙌 Acknowledgments

Special thanks to the Italian researcher who provided the dataset used in this study. This work was inspired by the need for better tools in early-stage cancer detection and diagnostic support.

