# Egyptian Currency Detection - Mobile Model

A lightweight currency classification model for Egyptian banknotes, trained entirely from scratch using PyTorch and exported as TorchScript Lite (`.ptl`) for mobile deployment.

## Classes

| Index | Class | Description |
|-------|-------|-------------|
| 0 | 1 | 1 EGP |
| 1 | 5 | 5 EGP |
| 2 | 10 | 10 EGP (old) |
| 3 | 10 (new) | 10 EGP (new) |
| 4 | 20 | 20 EGP (old) |
| 5 | 20 (new) | 20 EGP (new) |
| 6 | 50 | 50 EGP |
| 7 | 100 | 100 EGP |
| 8 | 200 | 200 EGP |

## Architecture

**CurrencyMobileNet** — A MobileNetV2-inspired architecture built entirely from scratch:

- **Depthwise separable convolutions** reduce computation by ~8-9x vs standard convolutions
- **Inverted residual blocks** with expansion ratios for efficient feature extraction
- **ReLU6 activations** are quantization-friendly for mobile deployment
- **Global average pooling** eliminates large FC layers
- **~2.2M parameters** (FP32: ~8.5 MB, Lite: ~8 MB)
- **No pretrained weights** — all layers are Kaiming-initialized and trained from scratch

### Why this architecture?

1. **Mobile-optimized**: Depthwise separable convolutions provide the best accuracy/FLOPS tradeoff
2. **Small model size**: Fits comfortably on mobile devices
3. **Fast inference**: ~15-30ms on modern mobile CPUs
4. **Quantization-ready**: ReLU6 and BatchNorm make it easy to quantize further

## Project Structure

```
├── config.py          # All hyperparameters and paths
├── dataset.py         # Dataset loading, validation, augmentation
├── model.py           # CurrencyMobileNet architecture
├── utils.py           # Metrics, early stopping, checkpointing, logging
├── train.py           # Training loop with AMP, scheduling, early stopping
├── evaluate.py        # Test set evaluation with full metrics
├── infer.py           # Single image / batch inference
├── export_ptl.py      # Export to TorchScript Lite (.ptl)
├── requirements.txt   # Python dependencies
├── README.md          # This file
├── data/
│   ├── train/         # Training images (class subfolders)
│   ├── valid/         # Validation images
│   └── test/          # Test images
└── outputs/
    ├── best_model.pth # Best checkpoint
    ├── model.ptl      # Exported mobile model
    ├── checkpoints/   # Periodic checkpoints
    └── logs/          # Training CSV logs
```

## Setup

```bash
pip install -r requirements.txt
```

## Commands

### 1. Validate Dataset

```bash
python dataset.py
```

### 2. Verify Model Architecture

```bash
python model.py
```

### 3. Train

```bash
# Default training (100 epochs, batch size 32, cosine LR)
python train.py

# Custom training
python train.py --epochs 50 --batch-size 64 --lr 0.005

# Resume from checkpoint
python train.py --resume outputs/checkpoints/checkpoint_epoch_10.pth
```

### 4. Evaluate on Test Set

```bash
# Evaluate best model on test set
python evaluate.py

# Evaluate on validation set with visualization
python evaluate.py --split valid --save-viz

# Evaluate specific checkpoint
python evaluate.py --checkpoint outputs/checkpoints/checkpoint_epoch_50.pth
```

### 5. Export to TorchScript Lite

```bash
# Export best model (with mobile optimizations)
python export_ptl.py

# Export without optimizations
python export_ptl.py --no-optimize

# Custom paths
python export_ptl.py --checkpoint outputs/best_model.pth --output outputs/model.ptl
```

### 6. Run Inference

```bash
# Single image
python infer.py --image path/to/currency_image.jpg

# Directory of images
python infer.py --image-dir path/to/images/

# Using exported mobile model
python infer.py --image path/to/image.jpg --model outputs/model.ptl

# Top-5 predictions
python infer.py --image path/to/image.jpg --top-k 5
```

## Training Features

- **From-scratch training**: All weights randomly initialized (Kaiming)
- **Class imbalance handling**: Weighted sampling + weighted loss
- **Label smoothing**: 0.1 for regularization
- **Data augmentation**: Random crop, flip, rotation, color jitter, perspective distortion, Gaussian blur, random erasing
- **Mixed precision (AMP)**: Faster training on GPU
- **Cosine annealing LR**: Smooth learning rate decay
- **Early stopping**: Patience of 15 epochs
- **Gradient clipping**: Max norm 5.0 for stability
- **Reproducible**: Fixed seeds across all random sources

## Robustness

The augmentation pipeline is designed to handle:
- Folded/wrinkled notes → RandomResizedCrop, RandomPerspective
- Partial occlusion → RandomErasing, RandomResizedCrop
- Varying lighting → ColorJitter (brightness, contrast, saturation)
- Scale variation → RandomResizedCrop with wide scale range
- Motion blur → GaussianBlur
- Worn/old currency → ColorJitter, aggressive augmentation
- Rotated notes → RandomRotation (±30°)
- Perspective distortion → RandomPerspective

## Mobile Integration (Android/Kotlin)

```kotlin
// Load model
val module = LiteModuleLoader.load(assetFilePath(this, "model.ptl"))

// Preprocess image (224x224, normalized)
val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
    bitmap,
    floatArrayOf(0.485f, 0.456f, 0.406f),  // mean
    floatArrayOf(0.229f, 0.224f, 0.225f)   // std
)

// Run inference
val output = module.forward(IValue.from(inputTensor)).toTensor()
val scores = output.dataAsFloatArray

// Get predicted class
val maxIdx = scores.indices.maxByOrNull { scores[it] } ?: 0
val classNames = arrayOf("1", "5", "10", "10 (new)", "20", "20 (new)", "50", "100", "200")
val prediction = classNames[maxIdx]
val confidence = softmax(scores)[maxIdx]
```

## Mobile Integration (iOS/Swift)

```swift
// Load model
let module = try! TorchModule(fileAtPath: modelPath)

// Preprocess image (224x224, normalized with mean/std)
let inputTensor = preprocess(image: uiImage)  // Your preprocessing function

// Run inference
let output = module.predict(input: inputTensor)

// Get predicted class
let classNames = ["1", "5", "10", "10 (new)", "20", "20 (new)", "50", "100", "200"]
let maxIdx = output.enumerated().max(by: { $0.element < $1.element })!.offset
let prediction = classNames[maxIdx]
```

## Evaluation Metrics

The evaluation script reports:
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class and macro-averaged
- **Recall**: Per-class and macro-averaged  
- **F1 Score**: Per-class and macro-averaged
- **mAP**: Mean Average Precision (classification)
- **Confusion Matrix**: Full NxN matrix
