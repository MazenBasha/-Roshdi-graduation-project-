from face_database import add_face, find_face

import cv2
import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras
from config.config import Config
from models.mobilefacenet_simple import create_mobilefacenet

# Build model with classification head
mobilefacenet = create_mobilefacenet(embedding_size=Config.EMBEDDING_SIZE)
inputs = keras.Input(shape=(Config.INPUT_SIZE, Config.INPUT_SIZE, 3))
embeddings = mobilefacenet(inputs, training=False)
logits = keras.layers.Dense(
    5749,
    activation="linear",
    use_bias=True,
    kernel_regularizer=keras.regularizers.l2(1e-4),
    name="classification"
)(embeddings)
model = keras.Model(inputs=inputs, outputs=logits)

# Load weights by name to handle architecture changes
with h5py.File('checkpoints/best_model.h5', 'r') as f:
    mf_group = f['model_weights/mobilefacenet']
    weight_names = mf_group.attrs.get('weight_names', [])
    layers_dict = {}
    for wn in weight_names:
        wn_str = wn.decode() if isinstance(wn, bytes) else wn
        layer_name = wn_str.split('/')[0]
        if layer_name not in layers_dict:
            layers_dict[layer_name] = []
        layers_dict[layer_name].append(wn_str)
    for layer_name, wnames in layers_dict.items():
        try:
            sub_layer = mobilefacenet.get_layer(layer_name)
            weights = [mf_group[wn][()] for wn in wnames]
            sub_layer.set_weights(weights)
        except Exception:
            pass
    # Load classification layer
    cls_group = f['model_weights/classification']
    cls_wnames = cls_group.attrs.get('weight_names', [])
    cls_weights = [cls_group[wn.decode() if isinstance(wn, bytes) else wn][()] for wn in cls_wnames]
    try:
        model.get_layer('classification').set_weights(cls_weights)
    except Exception:
        pass
print("Weights loaded successfully.")

# Extract embedding model
embedding_model = keras.Model(inputs=inputs, outputs=embeddings)

# Camera setup
cap = cv2.VideoCapture(0)
print("Camera started. Press 'q' to quit.")


while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Preprocess frame
    img = cv2.resize(frame, (Config.INPUT_SIZE, Config.INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    # Model inference
    embedding = embedding_model.predict(img)[0]

    # Find face in database
    name, dist, _ = find_face(embedding)

    if name:
        text = f"Recognized: {name} ({dist:.2f})"
        color = (0,0,0)  # Black
    else:
        text = "Unknown"
        color = (0,0,255)

    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
    cv2.imshow('Camera Test', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Register new face
        new_name = input("Enter name for new face: ")
        add_face(new_name, embedding)
        print(f"Registered {new_name}")
cap.release()
cv2.destroyAllWindows()
