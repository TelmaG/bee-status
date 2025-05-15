#!/usr/bin/env python3
"""
Audio Classification using YAMNet and Custom Models
This module provides tools for audio classification using pre-trained and custom models.
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import librosa
import resampy
import soundfile as sf
import tensorflow as tf
from tensorflow.keras.models import load_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


@dataclass(frozen=True)
class YAMNetParams:
    """Parameters for YAMNet model."""
    sample_rate: float = 16000.0
    stft_window_seconds: float = 0.025
    stft_hop_seconds: float = 0.010
    mel_bands: int = 64
    mel_min_hz: float = 125.0
    mel_max_hz: float = 7500.0
    log_offset: float = 0.001
    patch_window_seconds: float = 0.96
    patch_hop_seconds: float = 0.48
    num_classes: int = 521
    conv_padding: str = 'same'
    batchnorm_center: bool = True
    batchnorm_scale: bool = False
    batchnorm_epsilon: float = 1e-4
    classifier_activation: str = 'sigmoid'
    tflite_compatible: bool = True

    @property
    def patch_frames(self) -> int:
        """Calculate number of frames per patch."""
        return int(round(self.patch_window_seconds / self.stft_hop_seconds))

    @property
    def patch_bands(self) -> int:
        """Get number of mel bands."""
        return self.mel_bands


@dataclass
class ModelConfig:
    """Configuration for models and paths."""
    yamnet_model_path: str = 'models/yamnet.h5'
    yamnet_classes_path: str = 'yamnet_class_map.csv'
    custom_model_path: str = 'custom_model/model.h5'
    custom_classes_path: str = 'custom_model/model.npy'
    output_dir: str = 'result'
    output_file: str = 'analysis.txt'
    
    # Model parameters
    window_length: int = 10  # seconds
    hop_length: int = 1  # seconds
    custom_weight_factor: float = 5.0  # Weighting factor for custom model predictions
    
    # Runtime cache
    yamnet_model: Optional[tf.keras.Model] = field(default=None, repr=False)
    custom_model: Optional[tf.keras.Model] = field(default=None, repr=False)
    yamnet_classes: Optional[List[str]] = field(default=None, repr=False)
    custom_classes: Optional[np.ndarray] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Ensure output directory exists."""
        os.makedirs(Path(self.output_dir), exist_ok=True)
    
    @property
    def output_path(self) -> str:
        """Get full path to output file."""
        return os.path.join(self.output_dir, self.output_file)


class AudioProcessor:
    """Process audio files for classification."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the audio processor with configuration."""
        self.config = config
        self.params = YAMNetParams()
        self._init_models()
    
    def _init_models(self) -> None:
        """Initialize models and load classes."""
        try:
            # Initialize YAMNet model
            if not self.config.yamnet_model:
                self.config.yamnet_model = self._load_yamnet_model()
                
            # Load YAMNet class names
            if not self.config.yamnet_classes:
                self.config.yamnet_classes = self._load_yamnet_classes()
            
            # Initialize custom model
            if not self.config.custom_model and os.path.exists(self.config.custom_model_path):
                self.config.custom_model = load_model(self.config.custom_model_path)
                
            # Load custom class names
            if not self.config.custom_classes and os.path.exists(self.config.custom_classes_path):
                self.config.custom_classes = np.load(self.config.custom_classes_path, allow_pickle=True)
                
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    def _load_yamnet_model(self) -> tf.keras.Model:
        """Load YAMNet model."""
        try:
            from yamnet import yamnet_frames_model
            model = yamnet_frames_model(self.params)
            model.load_weights(self.config.yamnet_model_path)
            return model
        except ImportError:
            logger.error("YAMNet module not found. Provide the correct path.")
            raise
        except Exception as e:
            logger.error(f"Failed to load YAMNet model: {e}")
            raise
    
    def _load_yamnet_classes(self) -> List[str]:
        """Load YAMNet class names."""
        try:
            from yamnet import class_names
            return class_names(self.config.yamnet_classes_path)
        except ImportError:
            logger.error("YAMNet module not found. Provide the correct path.")
            raise
        except Exception as e:
            logger.error(f"Failed to load YAMNet classes: {e}")
            raise
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load and preprocess audio file."""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Audio file not found: {file_path}")
                
            # Load audio data
            wav_data, sr = sf.read(file_path, dtype=np.int16)
            
            # Validate audio data
            if wav_data.dtype != np.int16:
                logger.warning(f"Converting audio from {wav_data.dtype} to int16")
                wav_data = wav_data.astype(np.int16)
            
            # Convert to float32 in range [-1.0, 1.0]
            waveform = wav_data / 32768.0
            waveform = waveform.astype('float32')
            
            # Convert stereo to mono if needed
            if len(waveform.shape) > 1:
                logger.info("Converting stereo audio to mono")
                waveform = np.mean(waveform, axis=1)
            
            # Resample if needed
            if sr != self.params.sample_rate:
                logger.info(f"Resampling audio from {sr}Hz to {self.params.sample_rate}Hz")
                waveform = resampy.resample(waveform, sr, self.params.sample_rate)
                sr = int(self.params.sample_rate)
            
            return waveform, sr
            
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            raise
    
    def process_audio_segments(self, waveform: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Process audio in segments."""
        segment_length_samples = int(sr * self.config.window_length)
        hop_length_samples = int(sr * self.config.hop_length)
        
        if segment_length_samples <= 0:
            raise ValueError(f"Invalid segment length: {self.config.window_length} seconds")
        
        segments_results = []
        
        # Process each segment
        for i in range(0, max(1, len(waveform) - segment_length_samples + 1), hop_length_samples):
            end_idx = min(i + segment_length_samples, len(waveform))
            window = waveform[i:end_idx]
            
            # Get YAMNet predictions
            yamnet_predictions = self._get_yamnet_predictions(window)
            
            # Get custom model predictions if available
            custom_predictions = self._get_custom_predictions(window) if self.config.custom_model else None
            
            # Combine predictions
            combined_results = self._combine_predictions(yamnet_predictions, custom_predictions)
            
            # Store results
            segment_result = {
                'yamnet_predictions': yamnet_predictions,
                'custom_predictions': custom_predictions,
                'combined_predictions': combined_results
            }
            
            segments_results.append(segment_result)
        
        return segments_results
    
    def _get_yamnet_predictions(self, audio_segment: np.ndarray) -> Dict[str, float]:
        """Get YAMNet predictions for an audio segment."""
        try:
            scores, embeddings, spectrogram = self.config.yamnet_model(audio_segment)
            prediction = np.mean(scores, axis=0)
            
            # Get top predictions
            top_indices = np.argsort(prediction)[::-1][:10]
            top_labels = [self.config.yamnet_classes[i] for i in top_indices]
            top_scores = prediction[top_indices]
            
            return {label: float(score) for label, score in zip(top_labels, top_scores)}
            
        except Exception as e:
            logger.error(f"Error in YAMNet prediction: {e}")
            return {}
    
    def _get_custom_predictions(self, audio_segment: np.ndarray) -> Dict[str, float]:
        """Get custom model predictions for an audio segment."""
        try:
            # Get YAMNet embeddings first
            embeddings = self.config.yamnet_model(audio_segment)[1]
            
            # Reshape embeddings for custom model
            embeddings_reshaped = np.reshape(embeddings, (embeddings.shape[0], -1))
            
            # Get predictions from custom model
            predictions = self.config.custom_model.predict(embeddings_reshaped, verbose=0)
            
            # Get top predictions
            top_indices = np.argsort(np.mean(predictions, axis=0))[::-1][:10]
            top_labels = self.config.custom_classes[top_indices]
            top_scores = np.mean(predictions[:, top_indices], axis=0)
            
            # Normalize scores
            total_score = np.sum(top_scores)
            if total_score > 0:
                top_scores = top_scores / total_score
            
            return {label: float(score) for label, score in zip(top_labels, top_scores)}
            
        except Exception as e:
            logger.error(f"Error in custom model prediction: {e}")
            return {}
    
    def _combine_predictions(
        self, 
        yamnet_predictions: Dict[str, float], 
        custom_predictions: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """Combine predictions from different models."""
        combined = {}
        
        # Add custom predictions with weighting if available
        if custom_predictions:
            for label, score in custom_predictions.items():
                combined[label] = score * self.config.custom_weight_factor
        
        # Add YAMNet predictions if not already present or if higher score
        for label, score in yamnet_predictions.items():
            if label not in combined or score > combined[label]:
                if label != 'Vehicle':  # Skip 'Vehicle' class as per original code
                    combined[label] = score
        
        return combined
    
    def aggregate_results(self, segments_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across all segments."""
        # Initialize aggregated predictions
        aggregated_predictions = {}
        
        # Collect all combined predictions
        for segment in segments_results:
            for label, score in segment['combined_predictions'].items():
                if label in aggregated_predictions:
                    # Keep maximum score across segments
                    aggregated_predictions[label] = max(aggregated_predictions[label], score)
                else:
                    aggregated_predictions[label] = score
        
        # Find dominant class and normalize scores
        if aggregated_predictions:
            # Get top 10 predictions
            sorted_predictions = sorted(
                aggregated_predictions.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            # Create a new dictionary with only the top 10 predictions
            top_predictions = {label: score for label, score in sorted_predictions}
            
            # Calculate total score for normalization
            total_score = sum(top_predictions.values())
            
            # Normalize to ensure all probabilities sum to 1 (softmax-like)
            if total_score > 0:
                normalized_predictions = {
                    label: score / total_score 
                    for label, score in top_predictions.items()
                }
            else:
                # If all scores are 0, assign equal probability
                normalized_predictions = {
                    label: 1.0 / len(top_predictions) if len(top_predictions) > 0 else 0.0
                    for label in top_predictions
                }
            
            # Find dominant label
            dominant_label = max(normalized_predictions.items(), key=lambda x: x[1])[0]
            dominant_score = normalized_predictions[dominant_label]
            dominant_score_percentage = round(dominant_score * 100)
            
            # Replace original predictions with normalized ones
            aggregated_predictions = normalized_predictions
        else:
            dominant_label = "Unknown"
            dominant_score = 0
            dominant_score_percentage = 0
        
        return {
            'aggregated_predictions': aggregated_predictions,
            'dominant_label': dominant_label,
            'dominant_score': dominant_score,
            'dominant_score_percentage': dominant_score_percentage
        }
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save classification results to file."""
        try:
            with open(self.config.output_path, 'w') as file:
                file.write("Final Classification:\n")
                file.write(f"Label: {results['dominant_label']}\n")
                file.write(f"Probability: {results['dominant_score_percentage']}%\n")
                
                # Add more detailed information
                file.write("\nTop Predictions:\n")
                sorted_predictions = sorted(
                    results['aggregated_predictions'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10]
                
                # Calculate total sum for softmax-like normalization
                total_score = sum(score for _, score in sorted_predictions)
                
                if total_score > 0:
                    for label, score in sorted_predictions:
                        # Normalize as a percentage of the total (softmax-like)
                        percentage = round((score / total_score) * 100)
                        file.write(f"{label}: {percentage}%\n")
                else:
                    # Fallback if all scores are zero
                    for label, _ in sorted_predictions:
                        file.write(f"{label}: 0%\n")
                
            logger.info(f"Results saved to {self.config.output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise


def main():
    """Main function to run audio classification."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Audio Classification')
    parser.add_argument('audio_file', type=str, help='Path to audio file for classification')
    parser.add_argument('--yamnet_model', type=str, default='models/yamnet.h5', 
                        help='Path to YAMNet model')
    parser.add_argument('--custom_model', type=str, default='custom_model/model.h5', 
                        help='Path to custom model')
    parser.add_argument('--output', type=str, default='result/analysis.txt', 
                        help='Path to output file')
    parser.add_argument('--window', type=int, default=10, 
                        help='Window length in seconds')
    parser.add_argument('--hop', type=int, default=1, 
                        help='Hop length in seconds')
    parser.add_argument('--debug', action='store_true', 
                        help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Parse output path
        output_dir = os.path.dirname(args.output)
        output_file = os.path.basename(args.output)
        
        # Create configuration
        config = ModelConfig(
            yamnet_model_path=args.yamnet_model,
            custom_model_path=args.custom_model,
            output_dir=output_dir,
            output_file=output_file,
            window_length=args.window,
            hop_length=args.hop
        )
        
        # Initialize processor
        processor = AudioProcessor(config)
        
        # Load and process audio
        logger.info(f"Processing audio file: {args.audio_file}")
        waveform, sr = processor.load_audio(args.audio_file)
        
        # Process audio segments
        logger.info("Processing audio segments...")
        segments_results = processor.process_audio_segments(waveform, sr)
        
        # Aggregate results
        logger.info("Aggregating results...")
        final_results = processor.aggregate_results(segments_results)
        
        # Save results
        processor.save_results(final_results)
        
        # Print final classification
        print("Final Classification:")
        print(f"Label: {final_results['dominant_label']}")
        print(f"Probability: {final_results['dominant_score_percentage']}%")
        
        # Print top predictions with softmax-normalized percentages
        print("\nTop Predictions (normalized to sum 100%):")
        sorted_predictions = sorted(
            final_results['aggregated_predictions'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for label, score in sorted_predictions:
            print(f"{label}: {round(score * 100)}%")
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    import sys
    main()