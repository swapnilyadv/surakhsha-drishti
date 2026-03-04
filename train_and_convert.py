import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    TimeDistributed, LSTM, Dense, Dropout, 
    Bidirectional, GlobalAveragePooling2D, Input
)
from tensorflow.keras.applications import MobileNetV2
import os

def build_model(sequence_length=24, img_size=(112, 112)):
    """Creates the MobileNetV2 + Bi-LSTM Architecture."""
    print("Building model architecture...")
    
    # Base Feature Extractor (CNN)
    mobilenet = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(img_size[0], img_size[1], 3)
    )
    mobilenet.trainable = False

    model = Sequential([
        Input(shape=(sequence_length, img_size[0], img_size[1], 3)),
        
        # Extract features from each of the 24 frames
        TimeDistributed(mobilenet),
        TimeDistributed(GlobalAveragePooling2D()),
        
        # Analyze temporal movement (The LSTM part)
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        
        # Classification
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def convert_to_tflite(keras_model_path, tflite_output_path):
    """
    Converts .keras to .tflite with SELECT_TF_OPS enabled.
    This is required for LSTM/Flex operations support on Pi.
    """
    print(f"Converting {keras_model_path} to TFLite...")
    
    # Load the trained model
    model = tf.keras.models.load_model(keras_model_path)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # CRUCIAL: Enable Flex Delegate for LSTM support
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite default ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow Flex ops.
    ]
    
    # Optional: Optimize for size
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    with open(tflite_output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✓ Successfully created: {tflite_output_path}")

if __name__ == "__main__":
    # Define Paths
    KERAS_PATH = 'best_violence_model.keras'
    TFLITE_PATH = 'violence_model.tflite'

    # 1. Build and Save the Keras Model
    # (In a real scenario, you would call model.fit() here with your dataset)
    model = build_model()
    model.save(KERAS_PATH)
    print(f"✓ Successfully created: {KERAS_PATH}")

    # 2. Convert to TFLite
    convert_to_tflite(KERAS_PATH, TFLITE_PATH)