import h5py
from models.mobilefacenet_simple import create_mobilefacenet
from config.config import Config
from tensorflow import keras
import numpy as np

# Create the model
model = create_mobilefacenet(embedding_size=Config.EMBEDDING_SIZE)
inputs = keras.Input(shape=(Config.INPUT_SIZE, Config.INPUT_SIZE, 3))
embeddings = model(inputs, training=False)
logits = keras.layers.Dense(1600, activation="linear", use_bias=True, kernel_regularizer=keras.regularizers.l2(1e-4), name="classification")(embeddings)
full_model = keras.Model(inputs=inputs, outputs=logits)

# Build it
dummy = np.random.randn(1, 112, 112, 3).astype(np.float32)
_ = full_model(dummy, training=False)

with h5py.File('checkpoints/best_model.h5', 'r') as f:
    mfg = f['model_weights/mobilefacenet']
    wmf = mfg.attrs.get('weight_names', [])
    
    # Group by layer
    by_layer = {}
    for w in wmf:
        ws = w.decode() if isinstance(w, bytes) else w
        parts = ws.split('/')
        layer = parts[0]
        if layer not in by_layer:
            by_layer[layer] = []
        by_layer[layer].append(ws)
    
    print(f"In h5 file, {len(by_layer)} unique layers:")
    for layer_name in sorted(by_layer.keys()):
        weight_count = len(by_layer[layer_name])
        print(f"  {layer_name}: {weight_count} weights")
    
    print(f"\nModel has {len(full_model.layers)} layers total")
    mfnet = full_model.get_layer('mobilefacenet')
    print(f"Mobilefacenet has {len(mfnet.layers)} sub-layers:")
    for i, layer in enumerate(mfnet.layers):
        print(f"  {i}: {layer.name}")
        if i >= 10:
            print("  ...")
            break
