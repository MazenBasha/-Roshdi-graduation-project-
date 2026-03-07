"""
Evaluation script for face identification (1:N).
Measures top-1, top-k accuracy, unknown detection, best threshold.
"""
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from config.config import Config

# Utility: Cosine similarity

def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Evaluation

def evaluate_identification(model_path, gallery_dir, probe_dir, threshold=0.5, k=5):
    # Load model
    model = keras.models.load_model(model_path)
    # Build gallery
    gallery_embeddings = []
    gallery_labels = []
    for person in sorted(tf.io.gfile.listdir(gallery_dir)):
        imgs = tf.io.gfile.listdir(f"{gallery_dir}/{person}")
        for img in imgs:
            img_path = f"{gallery_dir}/{person}/{img}"
            image = tf.io.read_file(img_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [Config.INPUT_SIZE, Config.INPUT_SIZE])
            image = image / 127.5 - 1.0
            emb = model.predict(np.expand_dims(image, 0))[0]
            gallery_embeddings.append(emb)
            gallery_labels.append(person)
    # Probe
    probe_embeddings = []
    probe_labels = []
    for person in sorted(tf.io.gfile.listdir(probe_dir)):
        imgs = tf.io.gfile.listdir(f"{probe_dir}/{person}")
        for img in imgs:
            img_path = f"{probe_dir}/{person}/{img}"
            image = tf.io.read_file(img_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [Config.INPUT_SIZE, Config.INPUT_SIZE])
            image = image / 127.5 - 1.0
            emb = model.predict(np.expand_dims(image, 0))[0]
            probe_embeddings.append(emb)
            probe_labels.append(person)
    # Identification
    top1 = 0
    topk = 0
    unknown = 0
    for emb, label in zip(probe_embeddings, probe_labels):
        sims = [cosine_similarity(emb, g_emb) for g_emb in gallery_embeddings]
        sorted_idx = np.argsort(sims)[::-1]
        top_k_labels = [gallery_labels[i] for i in sorted_idx[:k]]
        best_score = sims[sorted_idx[0]]
        best_label = gallery_labels[sorted_idx[0]]
        if best_score < threshold:
            unknown += 1
        else:
            if best_label == label:
                top1 += 1
            if label in top_k_labels:
                topk += 1
    n = len(probe_labels)
    print(f"Top-1 accuracy: {top1/n:.4f}")
    print(f"Top-{k} accuracy: {topk/n:.4f}")
    print(f"Unknown detection rate: {unknown/n:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--gallery', type=str, required=True)
    parser.add_argument('--probe', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()
    evaluate_identification(args.model, args.gallery, args.probe, args.threshold, args.k)
