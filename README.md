# YOLO Demo Project

AI.SEE Assessment - YOLO Model Inference: PyTorch and ONNX Comparison

## 📋 Overview

This project demonstrates object detection using YOLO (You Only Look Once) models. It provides:
1. **PyTorch YOLO inference** on images using Ultralytics YOLO11
2. **Model export** from PyTorch to ONNX format
3. **ONNX inference** using ONNX Runtime for optimized deployment

The project compares PyTorch and ONNX inference approaches, showcasing how to convert and deploy YOLO models for production use.

## 🗂️ Project Structure

```
yolo_demo/
├── model/                      # Model files directory
│   ├── yolo11n.pt             # YOLO11 PyTorch model (pre-trained)
│   └── yolo11n.onnx           # YOLO11 ONNX model (generated)
│
├── script/                     # Python scripts directory
│   ├── inference.py           # PyTorch inference script
│   ├── export_onnx.py         # ONNX export script
│   └── inference_onnx.py      # ONNX inference script
│
├── static/                     # Static assets directory
│   ├── image.jpg              # Input test image
│   ├── output.jpg             # PyTorch inference output
│   └── output_onnx.jpg        # ONNX inference output
│
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore file
└── README.md                  # This file
```

## 🚀 Installation Guide

### Prerequisites

- **Python 3.8 or higher** (Python 3.9+ recommended)
- **pip** (Python package manager)
- **Git** (optional, for cloning the repository)

### Step-by-Step Installation

#### 1. Clone or Download the Project

If you have the project in a Git repository:
```bash
git clone <repository-url>
cd yolo_demo
```

Or simply navigate to the project directory if you already have it.

#### 2. Create a Virtual Environment (Recommended)

Creating a virtual environment isolates project dependencies and prevents conflicts:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On Linux/Mac:
source .venv/bin/activate

# On Windows:
# .venv\Scripts\activate
```

You should see `(.venv)` prefix in your terminal prompt when activated.

#### 3. Install Dependencies

Install all required packages from `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Alternative Installation (if requirements.txt fails):**

You can install packages individually:

```bash
# Core PyTorch (CPU version)
pip install torch torchvision torchaudio

# YOLO library
pip install ultralytics

# ONNX support
pip install onnx onnxruntime

# Image processing
pip install opencv-python numpy matplotlib Pillow

# Utilities
pip install tqdm
```

#### 4. Verify Installation

Test that key packages are installed correctly:

```bash
python3 -c "import torch; import ultralytics; import onnxruntime; print('All packages installed successfully!')"
```

#### 5. Download Model (if needed)

The project expects `yolo11n.pt` in the `model/` directory. If it's missing, the Ultralytics library will automatically download it on first use, or you can download it manually.

#### 6. Prepare Test Image

Ensure you have a test image named `image.jpg` in the `static/` directory. You can use any image file and rename it to `image.jpg`.

## 📖 How the Project Works

### Architecture Overview

The project implements a complete YOLO inference pipeline with two approaches:

1. **PyTorch Inference**: Direct inference using Ultralytics YOLO library
2. **ONNX Inference**: Optimized inference using ONNX Runtime after model conversion

### Workflow

```
Input Image (static/image.jpg)
    ↓
┌─────────────────────────────────────┐
│  1. PyTorch Inference              │
│     - Load yolo11n.pt              │
│     - Run inference                │
│     - Post-process results          │
│     - Save output.jpg              │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  2. Export to ONNX                 │
│     - Convert PyTorch → ONNX       │
│     - Optimize graph                │
│     - Save yolo11n.onnx            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  3. ONNX Inference                  │
│     - Load yolo11n.onnx            │
│     - Preprocess image              │
│     - Run inference                │
│     - Decode outputs                │
│     - Save output_onnx.jpg         │
└─────────────────────────────────────┘
```

### Key Components

#### 1. PyTorch Inference (`script/inference.py`)

**How it works:**
- Loads the YOLO11 PyTorch model (`yolo11n.pt`)
- Processes the input image through the model
- The Ultralytics library handles:
  - Image preprocessing (resize, normalization)
  - Model inference
  - Post-processing (NMS, confidence filtering)
  - Result visualization
- Outputs bounding boxes, class labels, and confidence scores
- Saves annotated image with detections

**Advantages:**
- Simple API, minimal code
- Automatic preprocessing/post-processing
- Built-in visualization

#### 2. ONNX Export (`script/export_onnx.py`)

**How it works:**
- Loads the PyTorch model
- Converts model to ONNX format:
  - Input shape: 640×640×3 (fixed)
  - Output format: Detection tensor
  - Graph simplification for optimization
- Saves optimized ONNX model

**Why ONNX?**
- Cross-platform compatibility
- Optimized inference speed
- Smaller model size
- Production deployment ready

#### 3. ONNX Inference (`script/inference_onnx.py`)

**How it works:**
- **Preprocessing:**
  - Loads image using OpenCV
  - Resizes to 640×640
  - Converts BGR to RGB
  - Normalizes pixel values (0-1 range)
  - Transposes to CHW format (Channel, Height, Width)
  - Adds batch dimension

- **Inference:**
  - Loads ONNX model with ONNX Runtime
  - Runs inference on preprocessed image
  - Gets raw output tensor

- **Post-processing:**
  - Decodes output tensor to bounding boxes
  - Applies confidence threshold (0.5)
  - Converts coordinates from model space to original image space
  - Draws bounding boxes and labels

**Advantages:**
- Faster inference (optimized runtime)
- Lower memory footprint
- Production-ready deployment

## 🎯 Usage

### Running Individual Scripts

#### 1. PyTorch Inference

Run inference using the PyTorch model:

```bash
python script/inference.py
```

**What it does:**
- Loads `model/yolo11n.pt`
- Processes `static/image.jpg`
- Prints detection details (class, confidence, bounding boxes)
- Saves annotated image to `static/output.jpg`

**Expected Output:**
```
Loading YOLO model from model/yolo11n.pt...
Running inference on static/image.jpg...

============================================================
Image: static/image.jpg
============================================================

Found 3 detection(s):

Detection 1:
  Class ID: 0
  Class Name: person
  Confidence: 0.8542 (85.42%)
  Bounding Box:
    Top-Left: (120.50, 80.30)
    Bottom-Right: (250.75, 400.20)
    Width: 130.25, Height: 319.90

...
```

#### 2. Export Model to ONNX

Convert the PyTorch model to ONNX format:

```bash
python script/export_onnx.py
```

**What it does:**
- Loads `model/yolo11n.pt`
- Converts to ONNX format with:
  - Fixed input size: 640×640
  - Simplified graph
  - ONNX opset version 12
- Saves `model/yolo11n.onnx`

**Note:** This only needs to be run once. The ONNX model will be reused for subsequent ONNX inference runs.

#### 3. ONNX Inference

Run inference using the ONNX model:

```bash
python script/inference_onnx.py
```

**What it does:**
- Loads `model/yolo11n.onnx` (must exist - run export first)
- Preprocesses `static/image.jpg`
- Runs inference with ONNX Runtime
- Applies confidence threshold (0.5)
- Draws bounding boxes and saves to `static/output_onnx.jpg`

**Expected Output:**
```
Class: 0, Confidence: 0.85, BBox: [120, 80, 250, 400]
Class: 2, Confidence: 0.72, BBox: [300, 150, 500, 350]
...
```

### Complete Workflow

Run all scripts in sequence:

```bash
# Step 1: PyTorch inference
python script/inference.py

# Step 2: Export to ONNX (only needed once)
python script/export_onnx.py

# Step 3: ONNX inference
python script/inference_onnx.py
```

## 📊 Understanding the Output

### Detection Format

Each detection includes:
- **Class ID**: Integer (0-79 for COCO dataset, 80 classes total)
- **Class Name**: Human-readable label (e.g., "person", "car", "dog")
- **Confidence Score**: Float between 0.0 and 1.0 (probability)
- **Bounding Box**: Coordinates `(x1, y1, x2, y2)` in pixels
  - `x1, y1`: Top-left corner
  - `x2, y2`: Bottom-right corner

### Output Images

- **`static/output.jpg`**: PyTorch inference result with colored bounding boxes and labels
- **`static/output_onnx.jpg`**: ONNX inference result with bounding boxes and class IDs

### COCO Classes

The model is trained on the COCO dataset with 80 object classes including:
- Person, bicycle, car, motorcycle, airplane, bus, train, truck
- Bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- Backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard
- And many more...

## 🔧 Configuration

### Adjustable Parameters

#### `script/inference_onnx.py`
```python
CONF_THRESHOLD = 0.5    # Minimum confidence for detection (0.0 - 1.0)
```

Lower values detect more objects but may include false positives.
Higher values are more conservative but may miss some objects.

#### `script/export_onnx.py`
```python
imgsz = 640            # Input image size (model input resolution)
dynamic = False        # Static input shape (True for dynamic shapes)
simplify = True        # Simplify ONNX graph for optimization
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### Issue: "Model file not found"
**Error:** `Error: Model file 'model/yolo11n.pt' not found!`

**Solution:**
- Ensure the `model/` directory exists
- Check that `yolo11n.pt` is in the `model/` directory
- The model will auto-download on first use if using Ultralytics

#### Issue: "Image file not found"
**Error:** `Error: Image file 'static/image.jpg' not found!`

**Solution:**
- Ensure the `static/` directory exists
- Place your test image in `static/` and name it `image.jpg`
- Or modify the script to use a different image path

#### Issue: "ONNX model not found"
**Error:** ONNX Runtime error when loading model

**Solution:**
- Run `python script/export_onnx.py` first to generate the ONNX model
- Ensure `model/yolo11n.onnx` exists before running ONNX inference

#### Issue: Import errors
**Error:** `ModuleNotFoundError: No module named 'ultralytics'`

**Solution:**
1. Activate your virtual environment: `source .venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Verify installation: `pip list | grep ultralytics`

#### Issue: CUDA/GPU errors
**Error:** CUDA-related errors (if you don't have GPU)

**Solution:**
- The project uses CPU by default
- If errors occur, ensure you installed CPU-only PyTorch:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```

#### Issue: No detections found
**Possible causes:**
- Image doesn't contain recognizable objects
- Confidence threshold too high
- Image preprocessing issues

**Solution:**
- Lower confidence threshold in `inference_onnx.py`
- Try a different image with clear objects
- Check that image loads correctly (print image shape)

#### Issue: Permission errors
**Error:** Permission denied when saving files

**Solution:**
- Check write permissions in `static/` directory
- Run with appropriate permissions: `chmod +w static/`

## 📦 Dependencies

### Core Dependencies

- **torch** (≥2.0.0): PyTorch deep learning framework
- **torchvision** (≥0.15.0): Computer vision utilities for PyTorch
- **torchaudio** (≥2.0.0): Audio processing for PyTorch
- **ultralytics** (≥8.0.0): YOLO implementation and model library

### ONNX Dependencies

- **onnx** (≥1.14.0): ONNX format support and conversion tools
- **onnxruntime** (≥1.15.0): ONNX Runtime inference engine

### Image Processing

- **opencv-python** (≥4.8.0): Computer vision and image processing
- **numpy** (≥1.24.0): Numerical operations and array handling
- **Pillow** (≥10.0.0): Image manipulation library

### Visualization

- **matplotlib** (≥3.7.0): Plotting and visualization

### Utilities

- **tqdm** (≥4.65.0): Progress bars for long operations

## 🎓 Learning Outcomes

This project demonstrates:

1. **Model Loading**: How to load pre-trained YOLO models
2. **Inference**: Running object detection on images
3. **Model Conversion**: Converting PyTorch models to ONNX format
4. **ONNX Runtime**: Using optimized inference engines
5. **Image Preprocessing**: Manual preprocessing for ONNX models
6. **Post-processing**: Decoding detection outputs and drawing results
7. **Production Deployment**: Preparing models for production use

## 📝 Notes

- **CPU Execution**: Scripts use CPU by default (no GPU required)
- **Model Size**: YOLO11n is a nano model (~6MB), optimized for speed
- **Input Size**: Models expect 640×640 input images
- **Output Format**: Detections use COCO dataset class labels
- **Performance**: ONNX Runtime typically provides faster inference than PyTorch

## 🔗 Additional Resources

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [ONNX Documentation](https://onnx.ai/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [COCO Dataset](https://cocodataset.org/)

## 📄 License

This is an assessment project for AI.SEE.

## 👤 Author

Created using Cursor AI Pro for the AI.SEE technical assessment.

---

**Last Updated**: December 2024
