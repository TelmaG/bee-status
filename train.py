#!/usr/bin/env python3
"""
Audio Classification System

This script trains a neural network model on audio data using YAMNet embeddings.
It extracts features from audio files and trains a classifier to recognize audio classes.

Usage: 
    python train.py <path_to_data> <model_name>
    
Requirements:
    - TensorFlow
    - librosa
    - numpy
    - pandas
    - scikit-learn
    - tqdm
    - YAMNet model and parameters
"""

import sys
import os
import numpy as np
import librosa
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from tqdm import tqdm
from tensorflow.keras import layers, Model, callbacks

# Import YAMNet
from yamnet import yamnet_frames_model
from params import Params

# Constants
YAMNET_PATH = "models/yamnet.h5"
CLASSES_PATH = "yamnet_class_map.csv"

def load_yamnet_classes(csv_path):
    """Load classes from YAMNet class map CSV file."""
    try:
        df = pd.read_csv(csv_path)
        return df["display_name"].values
    except FileNotFoundError:
        print(f"Class map file not found: {csv_path}. Creating a new one.")
        # Create an empty DataFrame if the file doesn't exist
        pd.DataFrame({"display_name": [], "index": [], "mid": []}).to_csv(csv_path, index=False)
        return np.array([])

def create_dataset(path):
    """
    Create a dataset from audio files in the specified path.
    
    Args:
        path: Path to the directory containing audio files organized in class folders
        
    Returns:
        samples: Numpy array of audio features
        labels: Numpy array of corresponding labels
    """
    samples, labels = [], []
    
    # Load YAMNet model for feature extraction
    print("Loading YAMNet model...")
    model = yamnet_frames_model(Params())
    model.load_weights(YAMNET_PATH)
    
    # Load or create classes mapping
    try:
        existing_classes_df = pd.read_csv(CLASSES_PATH)
    except FileNotFoundError:
        # Create empty DataFrame if file doesn't exist
        existing_classes_df = pd.DataFrame({"display_name": [], "index": [], "mid": []})
        existing_classes_df.to_csv(CLASSES_PATH, index=False)

    # Extract existing classes
    existing_classes_set = set(existing_classes_df['display_name'])
    
    # Find new classes
    new_classes = []
    for cls in sorted(os.listdir(path)):
        if os.path.isdir(os.path.join(path, cls)) and cls not in existing_classes_set:
            new_classes.append(cls)
    
    # Append new classes to the existing classes dataframe
    if new_classes:
        print(f"Adding {len(new_classes)} new classes: {', '.join(new_classes)}")
        new_classes_df = pd.DataFrame({
            'display_name': new_classes, 
            'index': [''] * len(new_classes), 
            'mid': [''] * len(new_classes)
        })
        updated_classes_df = pd.concat([existing_classes_df, new_classes_df], ignore_index=True)
        updated_classes_df.to_csv(CLASSES_PATH, index=False)
    
    # Extract features from each audio file
    for cls in sorted(os.listdir(path)):
        class_path = os.path.join(path, cls)
        if not os.path.isdir(class_path):
            continue
            
        print(f"Processing class: {cls}")
        audio_files = os.listdir(class_path)
        
        for sound in tqdm(audio_files, desc=f"Processing {cls}"):
            try:
                # Load audio file
                audio_path = os.path.join(class_path, sound)
                wav, _ = librosa.load(audio_path, sr=16000, mono=True)
                wav = wav.astype(np.float32)
                
                if len(wav) == 0:
                    print(f"Warning: Empty audio file: {audio_path}")
                    continue
                
                # Extract embeddings using YAMNet
                _, embeddings, _ = model(wav)
                
                # Store each embedding frame with its label
                for embedding in embeddings:
                    samples.append(embedding.numpy())
                    labels.append(cls)
                    
            except Exception as e:
                print(f"Error processing {sound}: {str(e)}")
    
    # Convert to numpy arrays
    if not samples:
        raise ValueError("No valid audio samples were processed!")
        
    samples = np.asarray(samples)
    labels = np.asarray(labels)
    
    print(f"Created dataset with {len(samples)} samples across {len(set(labels))} classes")
    return samples, labels

def generate_model(num_classes,
                  num_hidden=1024,
                  activation='relu',
                  final_activation='softmax',
                  regularization=0.01,
                  num_extra_layers=1,
                  hidden_layer_size=512,
                  dropout_rate=0.3):
    """
    Generate a neural network model for audio classification.
    
    Args:
        num_classes: Number of output classes
        num_hidden: Size of the first hidden layer
        activation: Activation function for hidden layers
        final_activation: Activation function for output layer
        regularization: L2 regularization factor
        num_extra_layers: Number of additional hidden layers
        hidden_layer_size: Size of additional hidden layers
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Keras Model object
    """
    # Input layer (YAMNet embeddings are 1024-dimensional)
    inputs = layers.Input(shape=(1024,))
    
    # First hidden layer with L2 regularization
    x = layers.Dense(
        num_hidden, 
        activation=activation,
        kernel_regularizer=tf.keras.regularizers.l2(regularization)
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Additional hidden layers
    for i in range(num_extra_layers):
        x = layers.Dense(
            hidden_layer_size // (i+1), 
            activation=activation,
            kernel_regularizer=tf.keras.regularizers.l2(regularization)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation=final_activation)(x)
    
    # Create and return model
    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_model(X, y, model_path,
               epochs=100,
               batch_size=32,
               learning_rate=0.001,
               num_hidden=1024,
               patience=10):
    """
    Train a model on the provided data.
    
    Args:
        X: Input features
        y: Target labels
        model_path: Path to save the model
        epochs: Maximum number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        num_hidden: Size of the first hidden layer
        patience: Early stopping patience
        
    Returns:
        Trained model
    """
    # Encode the labels (one-hot encoding)
    encoder = LabelBinarizer()
    encoded_labels = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)
    
    print(f"Training model with {num_classes} classes: {', '.join(encoder.classes_)}")
    
    # Create model
    model = generate_model(
        num_classes=num_classes,
        num_hidden=num_hidden,
        activation='relu',
        final_activation='softmax',
        regularization=0.01,
        num_extra_layers=1,
        hidden_layer_size=512,
        dropout_rate=0.3
    )
    
    # Print model summary
    model.summary()
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    # Create callbacks
    model_checkpoint = callbacks.ModelCheckpoint(
        filepath=f"{model_path}_best.h5",
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    tensorboard = callbacks.TensorBoard(
        log_dir=f"logs/{model_path}",
        histogram_freq=1
    )
    
    # Train the model
    history = model.fit(
        X, encoded_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[model_checkpoint, early_stopping, reduce_lr, tensorboard],
        verbose=1
    )
    
    # Save the model and class names
    model.save(f"{model_path}.h5")
    np.save(f"{model_path}_classes.npy", encoder.classes_)
    
    # Save training history
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(f"{model_path}_history.csv", index=False)
    
    print(f"Model saved as {model_path}.h5")
    print(f"Class names saved as {model_path}_classes.npy")
    
    return model

def main():
    """Main function to run the script."""
    # Check command-line arguments
    if len(sys.argv) < 3:
        print("Usage: train.py <path_to_data> <model_name>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    model_name = sys.argv[2]
    
    print(f"Data path: {data_path}")
    print(f"Model name: {model_name}")
    
    # Create dataset
    try:
        samples, labels = create_dataset(data_path)
    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        sys.exit(1)
    
    # Shuffle the data for better training
    samples, labels = shuffle(samples, labels, random_state=42)
    
    # Train model
    try:
        model = train_model(samples, labels, model_name)
        print(f"Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()