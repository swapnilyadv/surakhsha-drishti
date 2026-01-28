#!/usr/bin/env python3
"""
Model Converter for Raspberry Pi
=================================
Converts the Keras model to TensorFlow Lite format for faster inference on Raspberry Pi.

Run this script on your computer (not on Pi) before deploying:
    python convert_model.py

This will create 'violence_model.tflite' which is optimized for Raspberry Pi.
"""

import os
import tensorflow as tf

# Configuration
KERAS_MODEL_PATH = 'best_violence_model.keras'
TFLITE_MODEL_PATH = 'violence_model.tflite'


def convert_model():
    """Convert Keras model to TFLite with optimizations."""
    
    print("="*60)
    print("   Model Converter for Raspberry Pi")
    print("="*60)
    
    # Check if Keras model exists
    if not os.path.exists(KERAS_MODEL_PATH):
        print(f"\n✗ ERROR: Model not found at '{KERAS_MODEL_PATH}'")
        return False
    
    print(f"\n1. Loading Keras model: {KERAS_MODEL_PATH}")
    model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    print(f"   Model loaded successfully!")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    
    # Convert to TFLite
    print(f"\n2. Converting to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply optimizations for Raspberry Pi
    print("   Applying optimizations:")
    print("   - Default optimization (quantization)")
    print("   - Float16 quantization for ARM CPUs")
    print("   - SELECT_TF_OPS for LSTM support")
    
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    # Enable TF ops for LSTM layers (required for Bidirectional LSTM)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    print(f"\n3. Saving TFLite model: {TFLITE_MODEL_PATH}")
    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
    
    # Get file sizes
    keras_size = os.path.getsize(KERAS_MODEL_PATH) / (1024 * 1024)
    tflite_size = os.path.getsize(TFLITE_MODEL_PATH) / (1024 * 1024)
    reduction = (1 - tflite_size / keras_size) * 100
    
    print(f"\n✓ Conversion complete!")
    print(f"\n   Original size: {keras_size:.2f} MB")
    print(f"   TFLite size:   {tflite_size:.2f} MB")
    print(f"   Size reduction: {reduction:.1f}%")
    
    print(f"\n4. Copy these files to your Raspberry Pi:")
    print(f"   - {TFLITE_MODEL_PATH}")
    print(f"   - violence_detector_pi.py")
    
    return True


if __name__ == '__main__':
    success = convert_model()
    
    if success:
        print("\n" + "="*60)
        print("   Next Steps:")
        print("="*60)
        print("""
1. Copy files to Raspberry Pi:
   scp violence_model.tflite raspberry_pi/violence_detector_pi.py pi@<PI_IP>:~/violence_detection/

2. On Raspberry Pi, install dependencies:
   pip install tflite-runtime opencv-python numpy

3. Run the detector:
   python violence_detector_pi.py
        """)
