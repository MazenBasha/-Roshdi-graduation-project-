"""
Visualization script for training results.
Shows loss and accuracy curves for the LFW 30-epoch training.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load training history
history_path = Path(__file__).parent / 'training_history.json'
with open(history_path, 'r') as f:
    history = json.load(f)

# Extract metrics
epochs = np.arange(1, len(history['loss']) + 1)
train_loss = history['loss']
val_loss = history['val_loss']
train_acc = np.array(history['accuracy']) * 100  # Convert to percentage
val_acc = np.array(history['val_accuracy']) * 100

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Loss curves
ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
ax1.axvline(x=25, color='g', linestyle='--', label='Best Epoch (25)', alpha=0.7)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('ArcFace Loss over Epochs', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Plot 2: Accuracy curves
ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
ax2.axvline(x=25, color='g', linestyle='--', label='Best Epoch (25)', alpha=0.7)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Classification Accuracy over Epochs', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

# Adjust layout and save
plt.tight_layout()
output_path = Path(__file__).parent / 'training_curves.png'
plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
print(f"✅ Training curves saved to: {output_path}")

# Also print summary statistics
print("\n" + "="*60)
print("TRAINING SUMMARY STATISTICS")
print("="*60)
print(f"Initial Loss:        {train_loss[0]:.4f} (train), {val_loss[0]:.4f} (val)")
print(f"Final Loss:          {train_loss[-1]:.4f} (train), {val_loss[-1]:.4f} (val)")
print(f"Best Val Loss:       {min(val_loss):.4f} (Epoch {np.argmin(val_loss) + 1})")
print(f"\nInitial Accuracy:    {train_acc[0]:.2f}% (train), {val_acc[0]:.2f}% (val)")
print(f"Final Accuracy:      {train_acc[-1]:.2f}% (train), {val_acc[-1]:.2f}% (val)")
print(f"Best Val Accuracy:   {max(val_acc):.2f}% (Epoch {np.argmax(val_acc) + 1})")
print(f"\nLoss Reduction:      {(1 - train_loss[-1]/train_loss[0])*100:.1f}%")
print(f"Accuracy Gain:       {train_acc[-1] - train_acc[0]:.2f}%")
print("="*60)
