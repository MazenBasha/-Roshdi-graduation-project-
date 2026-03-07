"""
Evaluation script for face verification (1:1).
Measures ROC, EER, TAR@FAR, best threshold, accuracy.
"""
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_curve, auc
from config.config import Config

# Utility: Cosine similarity

def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Evaluation

def evaluate_verification(model_path, pairs_path, batch_size=32):
    # Load model
    model = keras.models.load_model(model_path)
    # Load pairs
    pairs = []
    with open(pairs_path, 'r') as f:
        for line in f:
            p1, p2, label = line.strip().split()
            pairs.append((p1, p2, int(label)))
    # Compute embeddings
    embeddings1 = []
    embeddings2 = []
    labels = []
    for p1, p2, label in pairs:
        img1 = tf.io.read_file(p1)
        img1 = tf.image.decode_jpeg(img1, channels=3)
        img1 = tf.image.resize(img1, [Config.INPUT_SIZE, Config.INPUT_SIZE])
        img1 = img1 / 127.5 - 1.0
        img2 = tf.io.read_file(p2)
        img2 = tf.image.decode_jpeg(img2, channels=3)
        img2 = tf.image.resize(img2, [Config.INPUT_SIZE, Config.INPUT_SIZE])
        img2 = img2 / 127.5 - 1.0
        emb1 = model.predict(np.expand_dims(img1, 0))[0]
        emb2 = model.predict(np.expand_dims(img2, 0))[0]
        embeddings1.append(emb1)
        embeddings2.append(emb2)
        labels.append(label)
    # Cosine similarities
    sims = [cosine_similarity(e1, e2) for e1, e2 in zip(embeddings1, embeddings2)]
    # ROC
    fpr, tpr, thresholds = roc_curve(labels, sims)
    roc_auc = auc(fpr, tpr)
    # EER
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fpr - fnr))]
    # Best threshold
    best_idx = np.argmax(tpr - fpr)
    best_thr = thresholds[best_idx]
    # TAR@FAR=0.01
    tar_at_far_01 = tpr[np.searchsorted(fpr, 0.01)] if np.any(fpr >= 0.01) else tpr[-1]
    # Accuracy
    preds = [int(s > best_thr) for s in sims]
    acc = np.mean([p == l for p, l in zip(preds, labels)])
    # Print
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"EER: {eer:.4f}")
    print(f"Best threshold: {best_thr:.4f}")
    print(f"TAR@FAR=0.01: {tar_at_far_01:.4f}")
    print(f"Accuracy: {acc:.4f}")
    # Save ROC curve
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.4f})')
    plt.xlabel('FAR')
    plt.ylabel('TAR')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('experiments/roc_curve.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--pairs', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    evaluate_verification(args.model, args.pairs, args.batch_size)
