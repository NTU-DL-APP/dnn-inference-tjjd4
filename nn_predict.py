import numpy as np
import json
import tensorflow as tf

# === Activation functions ===
def relu(x):
    # Stable ReLU implementation
    return np.maximum(0.0, x)

def softmax(x):
    # Numerically stable softmax
    x = x.astype(np.float64)  # Use float64 for better numerical stability
    x = x - np.max(x, axis=-1, keepdims=True)  # For numerical stability
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    # Ensure inputs are float64 for numerical stability
    x = x.astype(np.float64)
    W = W.astype(np.float64)
    b = b.astype(np.float64)
    
    # Clip weights and biases to reasonable ranges
    W = np.clip(W, -5.0, 5.0)
    b = np.clip(b, -5.0, 5.0)
    
    # Process in small batches to avoid numerical issues
    batch_size = 32
    output = np.zeros((x.shape[0], W.shape[1]), dtype=np.float64)
    
    for i in range(0, x.shape[0], batch_size):
        batch = x[i:i+batch_size]
        
        # Scale inputs to prevent overflow
        max_vals = np.max(np.abs(batch), axis=1, keepdims=True)
        scale = np.maximum(max_vals, 1.0)
        scaled_batch = batch / scale
        
        # Process each sample in the batch individually
        for j in range(batch.shape[0]):
            try:
                # Process with error handling
                result = np.dot(scaled_batch[j], W) * scale[j]
                output[i + j] = result + b
            except FloatingPointError:
                # Fallback for numerical instability
                output[i + j] = np.dot(batch[j], W) + b
    
    # Clip outputs to prevent extreme values
    output = np.clip(output, -1e4, 1e4)
    return output

# === Conv2D layer ===
def conv2d(x, W, b, strides=(1, 1), padding='valid'):
    # Convert inputs to TensorFlow tensors
    x_tf = tf.constant(x)
    W_tf = tf.constant(W)
    b_tf = tf.constant(b)
    
    # Perform convolution using TensorFlow
    output = tf.nn.conv2d(
        x_tf, 
        W_tf, 
        strides=[1, strides[0], strides[1], 1], 
        padding=padding.upper()
    )
    output = tf.nn.bias_add(output, b_tf)
    
    # Convert back to numpy and return
    return output.numpy()

# === MaxPooling2D layer ===
def max_pooling2d(x, pool_size=(2, 2), strides=(2, 2)):
    # Convert input to TensorFlow tensor
    x_tf = tf.constant(x)
    
    # Perform max pooling using TensorFlow
    output = tf.nn.max_pool2d(
        x_tf,
        ksize=[1, pool_size[0], pool_size[1], 1],
        strides=[1, strides[0], strides[1], 1],
        padding='VALID'
    )
    
    # Convert back to numpy and return
    return output.numpy()

def nn_forward_h5(model_arch, weights, data):
    x = data
    
    # Clip the initial input to reasonable range
    if np.issubdtype(x.dtype, np.floating):
        x = np.clip(x, -1e4, 1e4)
    
    for layer in model_arch:
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer.get('weights', [])
        
        if ltype == "InputLayer":
            continue  # Input layer doesn't need processing
            
        elif ltype == "Conv2D":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            strides = tuple(cfg.get('strides', (1, 1)))
            padding = cfg.get('padding', 'valid').lower()
            x = conv2d(x, W, b, strides=strides, padding=padding)
            if cfg.get('activation') == 'relu':
                x = relu(x)
                
        elif ltype == "MaxPooling2D":
            pool_size = tuple(cfg.get('pool_size', (2, 2)))
            strides = tuple(cfg.get('strides', pool_size))
            x = max_pooling2d(x, pool_size=pool_size, strides=strides)
            
        elif ltype == "Flatten":
            x = flatten(x)
            
        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            if cfg.get("activation") == "relu":
                x = relu(x)
            elif cfg.get("activation") == "softmax":
                x = softmax(x)
    
    return x


# You are free to replace nn_forward_h5() with your own implementation 
def nn_inference(model_arch, weights, data):
    # Ensure input has the correct shape (batch_size, 28, 28, 1)
    if len(data.shape) == 2:  # If input is (batch_size, 784)
        data = data.reshape(-1, 28, 28, 1)
    elif len(data.shape) == 3:  # If input is (batch_size, 28, 28)
        data = np.expand_dims(data, -1)
        
    # Ensure data type is float32
    data = data.astype(np.float32)
    
    # Run inference
    return nn_forward_h5(model_arch, weights, data)
    
