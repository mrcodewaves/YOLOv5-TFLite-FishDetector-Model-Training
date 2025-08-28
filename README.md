# YOLOv5-TFLite-FishDetector-Model-Training

This repository contains the **dataset engineering and model training pipeline** for the AI engine behind the [FishSpeciesDetector Android app](https://github.com/exxxius/FishSpeciesDetector).  
It documents the full lifecycle: **data collection → dataset creation → training & benchmarking → distillation → mobile deployment**.  

---

## 🎓 Academic Context

- **Author**: Mehdi (Elijah) Rahimi  
- **Created**: May 25, 2023  
- **Capstone Project**: Course C964 at Western Governors University  
- **Instructor**: Dr. Charles Paddock, Ph.D.  

This project was recognized for its potential applications in **environmental conservation and fisheries management**.

---

## 🌍 Motivation & Purpose

Correct identification of salmonid species is critical for:  

- **Fisheries Management**: Distinguishing between regulated species helps enforce size and catch rules.  
- **Conservation**: Prevents over-harvest of endangered runs.  
- **Research**: Enables population monitoring and migration studies.  
- **Community**: Equips anglers, students, and citizen scientists with AI tools.  

By developing a **lightweight, mobile-friendly fish detector**, this project makes real-time species recognition possible in the field without internet access.

---

## 📊 Dataset Creation

- **Target Classes**: Salmonidae species (Chinook, Coho, Sockeye, Pink, Chum, Steelhead).
- **Sources**:  
  - Google & Bing image searches  
  - Public angler/conservation photos  
  - Private fishing Facebook groups (with permission)  
  - Fisheries and environmental datasets  
- **Annotation**:  
  - Bounding boxes + species labels  
  - Tools used: LabelImg, CVAT  
- **Preprocessing**:  
  - Images normalized and resized (416×416, 640×640 for some runs)  
  - Augmentations: horizontal flips, color jitter, random crops  

Dataset grew to **thousands of annotated images** for robust model training.  

---

## ⚙️ Project Structure

- `main.py`: CLI automation for cloning YOLOv5, installing dependencies, training, testing, and exporting.  
- `train.py`: Training script with configurable parameters (image size, batch size, epochs).  
- `detect.py`: Run inference on test images.  
- `export.py`: Convert YOLOv5 PyTorch weights → ONNX → TensorFlow → TensorFlow Lite.  

---

## 🧠 Training Pipeline

The model was developed through **iterative transfer learning and distillation**:

1. **Baseline Models**  
   - SSD Inception  
   - MobileNet V2  
   - YOLOv8  
   Benchmarked for speed vs. accuracy.  

2. **YOLOv5 Training**  
   - Pretrained weights fine-tuned on custom dataset  
   - Epochs: 50–100 (depending on experiments)  
   - Input size: 416×416 and 640×640  
   - Metrics: mAP, precision, recall  

3. **Distillation & Conversion**  
   - YOLOv5 → PyTorch → ONNX → TensorFlow → TFLite  
   - Quantization & pruning applied for edge devices  
   - Final model size: **165 MB** (vs >1 GB in PyTorch)  
   - Inference speed: ~50–70ms on mid-range Android phones  

---

## 📈 Results

### Model Size Comparison

| Model      | Framework        | Size    | Notes |
|------------|------------------|---------|-------|
| YOLOv8     | PyTorch          | ~4 GB   | Very high accuracy, but too large for deployment |
| YOLOv5     | PyTorch          | ~1 GB   | Balanced model, still too large for mobile |
| **TFLite** | TensorFlow Lite  | **165 MB** | Optimized final model for Android, real-time capable |

---

### Training Metrics & Visualizations

#### Training Results
<p align="center">
  <img src="https://github.com/exxxius/YOLOv5-TFLite-FishDetector-Model-Training/raw/master/latest-training-results/results.png" width="70%" alt="Training Results">
</p>

#### Precision-Recall Curve
<p align="center">
  <img src="https://github.com/exxxius/YOLOv5-TFLite-FishDetector-Model-Training/raw/master/latest-training-results/PR_curve.png" width="70%" alt="PR Curve">
</p>

#### Confusion Matrix
<p align="center">
  <img src="https://github.com/exxxius/YOLOv5-TFLite-FishDetector-Model-Training/raw/master/latest-training-results/confusion_matrix.png" width="70%" alt="Confusion Matrix">
</p>

#### F1 Curve
<p align="center">
  <img src="https://github.com/exxxius/YOLOv5-TFLite-FishDetector-Model-Training/raw/master/latest-training-results/F1_curve.png" width="70%" alt="F1 Curve">
</p>

#### Label Distribution
<p align="center">
  <img src="https://github.com/exxxius/YOLOv5-TFLite-FishDetector-Model-Training/raw/master/latest-training-results/labels.jpg" width="70%" alt="Label Distribution">
</p>

---

## ▶️ How to Train & Export

1. Clone this repo:
   ```sh
   git clone https://github.com/exxxius/YOLOv5-TFLite-FishDetector-Model-Training.git
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Prepare your dataset:
   - Place images & labels under `/data/train`, `/data/val`, `/data/test`.  
   - Update `data.yaml` with class names.  

4. Train the model:
   ```sh
   python train.py --data data.yaml --img 640 --batch 16 --epochs 100 --weights yolov5s.pt
   ```

5. Export to TensorFlow Lite:
   ```sh
   python export.py --weights runs/train/exp/weights/best.pt --include tflite
   ```

---

## 🔮 Future Work

- Expand dataset to additional freshwater and saltwater species.  
- Improve robustness in low-light / underwater images.  
- Add semi-supervised learning for faster dataset growth.  
- Explore transformer-based detection models (DETR, YOLO-NAS).  

---

## 🧑‍💻 Skills Demonstrated

- Dataset engineering & annotation  
- Transfer learning & model benchmarking  
- YOLOv5 fine-tuning  
- Model conversion (PyTorch → TFLite)  
- Edge optimization (quantization, pruning)  
- Applied AI in environmental conservation  

---

## 📜 License

MIT License – see `LICENSE`.

---

## 📬 Contact

For professional inquiries, please connect via GitHub profile:  
[https://github.com/exxxius](https://github.com/exxxius)  
