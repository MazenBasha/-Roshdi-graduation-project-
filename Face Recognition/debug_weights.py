import h5py
import tensorflow as tf
from tensorflow import keras
from models.mobilefacenet_simple import create_mobilefacenet
import numpy as np
from config.config import Config

# Create the model structure
model = create_mobilefacenet(embedding_size=Config.EMBEDDING_SIZE)
inputs = keras.Input(shape=(Config.INPUT_SIZE, Config.INPUT_SIZE, 3))
embeddings = model(inputs, training=False)
logits = keras.layers.Dense(1600, activation="linear", use_bias=True, kernel_regularizer=keras.regularizers.l2(1e-4), name="classification")(embeddings)
full_model = keras.Model(inputs=inputs, outputs=logits)

# Build it
dummy_input = np.random.randn(1, Config.INPUT_SIZE, Config.INPUT_SIZE, 3).astype(np.float32)
_ = full_model(dummy_input, training=False)

print("Model structure:")
full_model.summary()

print("\n\nModel layers:")
for layer in full_model.layers:
    print(f"  {layer.name}: {type(layer).__name__}")
    if hasattr(layer, 'layers'):
        print(f"    Sub-layers: {[l.name for l in layer.layers[:3]]}")

print("\n\nWeight details:")
with h5py.File('checkpoints/best_model.h5', 'r') as f:
    if 'model_weights' in f:
        layer_names = f['model_weights'].attrs.get('layer_names', [])
        print(f"Layers in h5 file: {[l.decode() if isinstance(l, bytes) else l for l in layer_names]}")
        
        # Get mobilefacenet weights
        mobilefacenet_layer = full_model.get_layer('mobilefacenet')
        print(f"\nMobilefacenet layer: {type(mobilefacenet_layer)}")
        print(f"Number of sub-layers: {len(mobilefacenet_layer.layers)}")
        print(f"Total trainable weights: {len(mobilefacenet_layer.weights)}")
