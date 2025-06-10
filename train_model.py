import tensorflow as tf
import numpy as np
import json
from utils.mnist_reader import load_mnist

MODEL_NAME = 'fashion_mnist' # Default extension is h5
TF_MODEL_PATH = f'model/{MODEL_NAME}.h5'
MODEL_WEIGHTS_PATH = f'model/{MODEL_NAME}.npz'
MODEL_ARCH_PATH = f'model/{MODEL_NAME}.json'

# === Step 0: Train and save model if not exists ===
import os
if not os.path.exists(TF_MODEL_PATH):
    print("🚀 Training new model...")

    x_train, y_train = load_mnist("data/fashion", kind="t10k")
    x_test, y_test = load_mnist("data/fashion", kind="t10k")

    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    model.save(TF_MODEL_PATH)
    print(f"✅ Model trained and saved to {TF_MODEL_PATH}")
else:
    print(f"📦 Found existing model at {TF_MODEL_PATH}, skipping training.")

# === Step 1: Load Keras .h5 model ===
model = tf.keras.models.load_model(TF_MODEL_PATH)

# === Step 2: Print and collect weights ===
params = {}
print("🔍 Extracting weights from model...\n")
for layer in model.layers:
    weights = layer.get_weights()
    if weights:
        print(f"Layer: {layer.name}")
        for i, w in enumerate(weights):
            param_name = f"{layer.name}_{i}"
            print(f"  {param_name}: shape={w.shape}")
            params[param_name] = w
        print()

# === Step 3: Save to .npz ===
np.savez(MODEL_WEIGHTS_PATH, **params)
print(f"✅ Saved all weights to {MODEL_WEIGHTS_PATH}")

# === Step 4: Reload and verify ===
print("\n🔁 Verifying loaded .npz weights...\n")
loaded = np.load(MODEL_WEIGHTS_PATH)

for key in loaded.files:
    print(f"{key}: shape={loaded[key].shape}")

# === Step 6: Extract architecture to JSON ===
arch = []
for layer in model.layers:
    config = layer.get_config()
    info = {
        "name": layer.name,
        "type": layer.__class__.__name__,
        "config": config,
        "weights": [f"{layer.name}_{i}" for i in range(len(layer.get_weights()))]
    }
    arch.append(info)

with open(MODEL_ARCH_PATH, "w") as f:
    json.dump(arch, f, indent=2)

print("✅ Architecture saved to model_architecture.json")