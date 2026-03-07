"""
Quick examples for using the improved training pipeline.

Copy and modify these examples for your experiments.
"""

# ============================================================================
# EXAMPLE 1: Quick Start with Balanced Configuration
# ============================================================================

"""
Command line:
python quickstart_training.py --config balanced --train_dir lfw_funneled --val_dir lfw_funneled
"""

# Or programmatically:
from training.train_improved import ImprovedTrainingPipeline

pipeline = ImprovedTrainingPipeline(
    train_dir="lfw_funneled",
    val_dir="lfw_funneled",
    output_dir="experiments",
    exp_name="my_first_improved",
    batch_size=32,
    num_epochs=50,
    learning_rate=0.0003,
    augmentation_strength="strong",
    embedding_dropout=0.2,
    l2_reg=0.0001,
    label_smoothing=0.1,
)

model, history = pipeline.train()


# ============================================================================
# EXAMPLE 2: Conservative Configuration (Reduce Overfitting)
# ============================================================================

"""
Command line:
python quickstart_training.py --config conservative --epochs 50
"""

pipeline = ImprovedTrainingPipeline(
    train_dir="lfw_funneled",
    val_dir="lfw_funneled",
    output_dir="experiments",
    exp_name="conservative_test",
    batch_size=32,
    num_epochs=50,
    learning_rate=0.0003,
    augmentation_strength="strong",
    embedding_dropout=0.3,     # More dropout
    l2_reg=0.0005,             # More regularization
    label_smoothing=0.15,      # More smoothing
    arcface_margin=0.45,       # Smaller margin
)

model, history = pipeline.train()


# ============================================================================
# EXAMPLE 3: Aggressive Configuration (High Accuracy)
# ============================================================================

"""
Command line:
python quickstart_training.py --config aggressive --epochs 100
"""

pipeline = ImprovedTrainingPipeline(
    train_dir="lfw_funneled",
    val_dir="lfw_funneled",
    output_dir="experiments",
    exp_name="aggressive_test",
    batch_size=64,             # Larger batch
    num_epochs=100,
    learning_rate=0.0005,      # Higher learning rate
    augmentation_strength="strong",
    embedding_dropout=0.1,     # Less dropout
    l2_reg=0.00001,            # Less regularization
    label_smoothing=0.05,      # Less smoothing
)

model, history = pipeline.train()


# ============================================================================
# EXAMPLE 4: Grid Search Over Hyperparameters
# ============================================================================

"""
Run multiple experiments with different hyperparameters
"""

import itertools

# Parameters to search
margins = [0.4, 0.45, 0.5]
dropouts = [0.1, 0.2, 0.3]
l2_regs = [0.00001, 0.0001, 0.001]

for margin, dropout, l2 in itertools.product(margins, dropouts, l2_regs):
    exp_name = f"grid_m{margin}_d{dropout}_l{l2}"

    pipeline = ImprovedTrainingPipeline(
        train_dir="lfw_funneled",
        val_dir="lfw_funneled",
        output_dir="experiments",
        exp_name=exp_name,
        batch_size=32,
        num_epochs=30,
        arcface_margin=margin,
        embedding_dropout=dropout,
        l2_reg=l2,
    )

    print(f"\n\nRunning: {exp_name}")
    model, history = pipeline.train()


# ============================================================================
# EXAMPLE 5: Evaluate on LFW Verification Pairs
# ============================================================================

from evaluation.evaluate_lfw import evaluate_on_lfw

results = evaluate_on_lfw(
    model_path="experiments/001_balanced/best_model.h5",
    lfw_dir="lfw_funneled",
    output_dir="evaluation_results",
)

print(f"AUC: {results['roc_auc']:.6f}")
print(f"EER: {results['eer']:.6f}")
print(f"Best Accuracy: {results['best_accuracy']:.6f}")


# ============================================================================
# EXAMPLE 6: Compare Multiple Experiments
# ============================================================================

from training.experiment_comparator import ExperimentComparator

comparator = ExperimentComparator(Path("experiments"))

# Print summary
comparator.print_summary()

# Generate report
report = comparator.generate_comparison_report(Path("experiments/COMPARISON_REPORT.md"))
print(report)

# Plot all training curves
comparator.plot_all_training_curves(Path("experiments/comparison_plots"))


# ============================================================================
# EXAMPLE 7: Programmatic Dataset Loading with Filtering
# ============================================================================

from data.dataset_loader_v2 import load_improved_dataset

# Load with filtering and balancing
train_dataset, train_stats, train_loader = load_improved_dataset(
    data_dir="lfw_funneled",
    img_size=112,
    batch_size=32,
    augment=True,
    augmentation_strength="strong",
    min_images_per_class=3,    # Filter out classes with <3 images
    balance_classes=True,      # Balance class distribution
)

print(f"Classes: {train_stats['num_classes']}")
print(f"Samples: {train_stats['num_samples']}")
print(f"Batches: {train_stats['num_batches']}")
print(f"Class weights: {train_stats['class_weights'][:5]}...")  # First 5


# ============================================================================
# EXAMPLE 8: Learning Rate Schedule Visualization
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from training.lr_schedules import create_learning_rate_schedule

# Create schedule
total_steps = 414 * 50  # batches_per_epoch * num_epochs
warmup_steps = 414 * 5

lr_schedule = create_learning_rate_schedule(
    strategy="cosine_warmup",
    initial_learning_rate=0.0003,
    total_steps=total_steps,
    warmup_steps=warmup_steps,
)

# Compute schedule values
lrs = [float(lr_schedule(step)) for step in range(total_steps)]
steps = np.arange(len(lrs))

# Plot
plt.figure(figsize=(12, 6))
plt.plot(steps, lrs)
plt.xlabel("Training Step")
plt.ylabel("Learning Rate")
plt.title("Cosine Annealing with Warmup")
plt.grid(alpha=0.3)
plt.savefig("lr_schedule.png")
print("Learning rate schedule saved to lr_schedule.png")


# ============================================================================
# EXAMPLE 9: Custom Augmentation Pipeline
# ============================================================================

from data.augmentations_v2 import FaceAugmentationPipeline
import tensorflow as tf

# Create custom augmentation pipeline
augmenter = FaceAugmentationPipeline(
    img_size=112,
    brightness_delta=0.15,
    contrast_range=(0.85, 1.15),
    saturation_range=(0.75, 1.25),
    blur_probability=0.25,
    crop_probability=0.6,
    zoom_range=0.2,
    rotation_probability=0.3,
    flip_probability=0.5,
    erasing_probability=0.25,
    color_jitter_probability=0.3,
)

# Load and augment an image
import numpy as np
from PIL import Image

img = Image.open("test_image.jpg")
img_array = np.array(img, dtype=np.float32) / 127.5 - 1.0

# Convert to tensor
img_tensor = tf.constant(img_array)

# Apply augmentations
augmented = augmenter.augment(img_tensor).numpy()

print(f"Original: {img_tensor.shape}, range: [{img_tensor.min():.2f}, {img_tensor.max():.2f}]")
print(f"Augmented: {augmented.shape}, range: [{augmented.min():.2f}, {augmented.max():.2f}]")


# ============================================================================
# EXAMPLE 10: Manual Model Evaluation with Face Verification
# ============================================================================

from evaluation.face_verification import FaceVerificationEvaluator, load_lfw_pairs, load_image_for_verification
import tensorflow as tf

# Load model
model = tf.keras.models.load_model(
    "experiments/001_balanced/best_model.h5",
    custom_objects={}
)

# Create evaluator
evaluator = FaceVerificationEvaluator(model, similarity_metric="cosine")

# Compute embeddings for a batch of images
images = np.random.randn(10, 112, 112, 3)
embeddings = evaluator.compute_embeddings(images, batch_size=2)
print(f"Computed embeddings: {embeddings.shape}")

# Compute similarity between two embeddings
sim = evaluator.compute_similarity(embeddings[0], embeddings[1])
print(f"Similarity between image 0 and 1: {sim:.4f}")


# ============================================================================
# EXAMPLE 11: Visualizing Training Results
# ============================================================================

import json
import numpy as np
import matplotlib.pyplot as plt

# Load training history
with open("experiments/001_balanced/training_history.json") as f:
    history = json.load(f)

# Create plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

epochs = range(1, len(history["loss"]) + 1)

# Loss curve
ax1.plot(epochs, history["loss"], "b-", label="Train", linewidth=2)
ax1.plot(epochs, history["val_loss"], "r-", label="Validation", linewidth=2)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training Loss")
ax1.legend()
ax1.grid(alpha=0.3)

# Accuracy curve
ax2.plot(epochs, np.array(history["accuracy"]) * 100, "b-", label="Train", linewidth=2)
ax2.plot(epochs, np.array(history["val_accuracy"]) * 100, "r-", label="Validation", linewidth=2)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Training Accuracy")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_results.png", dpi=150)
print("Training results saved to training_results.png")


# ============================================================================
# EXAMPLE 12: Export Experiment Summary
# ============================================================================

from training.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker(Path("experiments"))

# List all experiments
experiments = tracker.list_experiments()

# Print each experiment
for exp in experiments:
    print(f"\n{exp['id']}:")
    print(f"  Path: {exp['path']}")
    print(f"  Config: {exp['config']}")
    print(f"  Metrics: {exp['metrics']}")

# Generate summary report
df = tracker.compare_experiments()
print("\nComparison Table:")
print(df)

# Export to CSV
df.to_csv("experiments_summary.csv", index=False)
print("\nExported to experiments_summary.csv")
