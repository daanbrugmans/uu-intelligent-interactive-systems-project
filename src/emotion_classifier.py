"""
Emotion Classifier using ConvNeXt-Tiny
======================================

This module provides real-time emotion classification from webcam frames
using a fine-tuned ConvNeXt-Tiny model trained on the EMOTIC dataset.

The model outputs Valence (positive/negative) and Arousal (energy level)
which are then mapped to 3 emotion categories: POSITIVE, NEUTRAL, NEGATIVE.

Architecture:
    Webcam Frame (BGR) → Preprocess → ConvNeXt-Tiny → [Valence, Arousal] → EmotionType
"""

import torch
import torchvision
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass
import logging
import time

from src.types import Emotion, EmotionType

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ClassifierConfig:
    """Configuration for the emotion classifier."""
    
    # Model settings
    weights_path: str = "checkpoints/best_head_ls.pt"
    input_size: int = 224
    
    # Valence/Arousal thresholds for emotion classification
    # These define the boundaries between POSITIVE/NEUTRAL/NEGATIVE
    # Based on EMOTIC dataset calibration
    valence_positive_threshold: float = 7.1   # Above this = POSITIVE
    valence_negative_threshold: float = 6.9   # Below this = NEGATIVE
    # Between 6.9 and 7.1 = NEUTRAL (narrow band)
    
    # Confidence calculation
    # Higher distance from thresholds = higher confidence
    confidence_scale: float = 2.0  # How quickly confidence increases with distance
    
    # Smoothing (to reduce jitter between frames)
    smoothing_enabled: bool = True
    smoothing_alpha: float = 0.3  # Higher = more responsive, lower = smoother
    
    # Performance
    inference_interval_ms: int = 100  # Minimum ms between inferences (10 FPS max)


# ============================================================================
# EMOTION CLASSIFIER
# ============================================================================

class EmotionClassifier:
    """
    Real-time emotion classifier using ConvNeXt-Tiny.
    
    This class:
        1. Loads the fine-tuned ConvNeXt model
        2. Processes webcam frames
        3. Outputs Valence/Arousal predictions
        4. Maps predictions to EmotionType (POSITIVE, NEUTRAL, NEGATIVE)
        5. Calculates confidence scores
    
    Usage:
        classifier = EmotionClassifier()
        
        # In your webcam loop:
        frame = camera.read()
        emotion, confidence = classifier.predict(frame)
        print(f"Detected: {emotion.emotion_type.name} ({confidence:.0%})")
    """
    
    def __init__(self, config: Optional[ClassifierConfig] = None):
        """
        Initialize the emotion classifier.
        
        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or ClassifierConfig()
        self.device = self._get_device()
        self.model = self._load_model()
        
        # Preprocessing tensors (on CPU, moved to device during inference)
        self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        
        # Smoothing state
        self._smoothed_valence: Optional[float] = None
        self._smoothed_arousal: Optional[float] = None
        
        # Rate limiting
        self._last_inference_time: float = 0.0
        self._last_result: Optional[Tuple[Emotion, float]] = None
        
        logger.info(f"✅ EmotionClassifier initialized on {self.device}")
        logger.info(f"   Thresholds: POSITIVE > {self.config.valence_positive_threshold}, "
                   f"NEGATIVE < {self.config.valence_negative_threshold}")
    
    def _get_device(self) -> torch.device:
        """Get the best available device."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    def _load_model(self) -> torch.nn.Module:
        """Load the fine-tuned ConvNeXt-Tiny model."""
        weights_path = Path(self.config.weights_path)
        
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {weights_path}. "
                "Please ensure the checkpoint file exists."
            )
        
        logger.info(f"Loading model from {weights_path}...")
        
        # Build model architecture
        model = torchvision.models.convnext_tiny(weights=None)
        
        # Load checkpoint
        ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
        state = ckpt['model_state'] if 'model_state' in ckpt else ckpt
        
        # Load feature extractor weights
        features_state = {
            k.replace('features.', ''): v 
            for k, v in state.items() 
            if k.startswith('features.')
        }
        model.features.load_state_dict(features_state)
        
        # Build custom regression head
        model.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.LayerNorm(768, eps=1e-6),
            torch.nn.Linear(768, 512),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 2)  # Output: [Valence, Arousal]
        )
        
        # Load classifier head weights
        classifier_state = {
            k.replace('classifier.', ''): v 
            for k, v in state.items() 
            if k.startswith('classifier.')
        }
        model.classifier.load_state_dict(classifier_state)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess a webcam frame for model inference.
        
        Args:
            frame: BGR image from OpenCV (H, W, 3)
        
        Returns:
            Preprocessed tensor (1, 3, 224, 224) on device
        """
        # Resize to model input size
        frame_resized = cv2.resize(
            frame, 
            (self.config.input_size, self.config.input_size),
            interpolation=cv2.INTER_CUBIC
        )
        
        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # To tensor and normalize
        tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
        tensor = (tensor - self.mean) / self.std
        
        # Add batch dimension and move to device
        return tensor.unsqueeze(0).to(self.device)
    
    def predict(self, frame: np.ndarray) -> Tuple[Emotion, float]:
        """
        Predict emotion from a webcam frame.
        
        Args:
            frame: BGR image from OpenCV
        
        Returns:
            Tuple of (Emotion object, confidence score 0-1)
        """
        # Rate limiting - return cached result if called too frequently
        current_time = time.time() * 1000  # ms
        if (self._last_result is not None and 
            current_time - self._last_inference_time < self.config.inference_interval_ms):
            return self._last_result
        
        # Preprocess
        input_tensor = self.preprocess(frame)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            pred = output.cpu().numpy()[0]
        
        # Convert from normalized [0,1] to valence/arousal [1,10]
        raw_valence = float(pred[0] * 9.0 + 1.0)
        raw_arousal = float(pred[1] * 9.0 + 1.0)
        
        # Apply smoothing
        if self.config.smoothing_enabled:
            valence, arousal = self._apply_smoothing(raw_valence, raw_arousal)
        else:
            valence, arousal = raw_valence, raw_arousal
        
        # Classify emotion and calculate confidence
        emotion_type, confidence = self._classify_valence(valence)
        
        # Create Emotion object
        emotion = Emotion(
            emotion_type=emotion_type,
            valence=valence,
            arousal=arousal
        )
        
        # Cache result
        self._last_result = (emotion, confidence)
        self._last_inference_time = current_time
        
        logger.debug(f"Predicted: {emotion_type.name} (V={valence:.2f}, A={arousal:.2f}, conf={confidence:.0%})")
        
        return emotion, confidence
    
    def _apply_smoothing(self, valence: float, arousal: float) -> Tuple[float, float]:
        """Apply exponential moving average smoothing."""
        alpha = self.config.smoothing_alpha
        
        if self._smoothed_valence is None:
            self._smoothed_valence = valence
            self._smoothed_arousal = arousal
        else:
            self._smoothed_valence = alpha * valence + (1 - alpha) * self._smoothed_valence
            self._smoothed_arousal = alpha * arousal + (1 - alpha) * self._smoothed_arousal
        
        return self._smoothed_valence, self._smoothed_arousal
    
    def _classify_valence(self, valence: float) -> Tuple[EmotionType, float]:
        """
        Classify valence into emotion type and calculate confidence.
        
        Args:
            valence: Valence score (1-10)
        
        Returns:
            Tuple of (EmotionType, confidence 0-1)
        """
        pos_thresh = self.config.valence_positive_threshold
        neg_thresh = self.config.valence_negative_threshold
        mid_point = (pos_thresh + neg_thresh) / 2
        
        if valence >= pos_thresh:
            emotion_type = EmotionType.POSITIVE
            # Confidence increases with distance from threshold
            distance = valence - pos_thresh
            confidence = min(1.0, 0.5 + distance * self.config.confidence_scale / 10)
            
        elif valence <= neg_thresh:
            emotion_type = EmotionType.NEGATIVE
            distance = neg_thresh - valence
            confidence = min(1.0, 0.5 + distance * self.config.confidence_scale / 10)
            
        else:
            emotion_type = EmotionType.NEUTRAL
            # Confidence is higher when closer to the middle of neutral zone
            distance_from_center = abs(valence - mid_point)
            max_distance = (pos_thresh - neg_thresh) / 2
            confidence = 0.5 + 0.3 * (1 - distance_from_center / max_distance)
        
        return emotion_type, confidence
    
    def reset_smoothing(self) -> None:
        """Reset smoothing state (call when starting a new session)."""
        self._smoothed_valence = None
        self._smoothed_arousal = None
        logger.debug("Smoothing state reset")
    
    def get_raw_prediction(self, frame: np.ndarray) -> Tuple[float, float]:
        """
        Get raw valence/arousal without classification.
        
        Useful for debugging or custom threshold tuning.
        
        Args:
            frame: BGR image from OpenCV
        
        Returns:
            Tuple of (valence 1-10, arousal 1-10)
        """
        input_tensor = self.preprocess(frame)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            pred = output.cpu().numpy()[0]
        
        valence = float(pred[0] * 9.0 + 1.0)
        arousal = float(pred[1] * 9.0 + 1.0)
        
        return valence, arousal


# ============================================================================
# THRESHOLD TESTING UTILITY
# ============================================================================

def test_thresholds_live(
    camera_id: int = 0,
    weights_path: str = "checkpoints/best_head_ls.pt"
) -> None:
    """
    Interactive tool to test and tune emotion thresholds on live webcam feed.
    
    Shows real-time valence/arousal values to help determine optimal thresholds.
    
    Controls:
        q - Quit
        r - Reset smoothing
        +/- - Adjust positive threshold
        [/] - Adjust negative threshold
    """
    config = ClassifierConfig(weights_path=weights_path)
    classifier = EmotionClassifier(config)
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_id}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("=" * 60)
    print("THRESHOLD TESTING MODE")
    print("=" * 60)
    print(f"Current thresholds:")
    print(f"  POSITIVE: valence > {config.valence_positive_threshold}")
    print(f"  NEGATIVE: valence < {config.valence_negative_threshold}")
    print(f"  NEUTRAL:  between the two")
    print()
    print("Controls:")
    print("  q - Quit")
    print("  r - Reset smoothing")
    print("  +/- - Adjust positive threshold")
    print("  [/] - Adjust negative threshold")
    print("=" * 60)
    
    # Color mapping
    colors = {
        EmotionType.POSITIVE: (0, 255, 0),   # Green
        EmotionType.NEUTRAL: (255, 255, 0),  # Cyan
        EmotionType.NEGATIVE: (0, 0, 255),   # Red
    }
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get prediction
        emotion, confidence = classifier.predict(frame)
        color = colors[emotion.emotion_type]
        
        # Draw overlay
        h, w = frame.shape[:2]
        
        # Background box
        cv2.rectangle(frame, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 200), color, 3)
        
        # Values
        cv2.putText(frame, f"Valence: {emotion.valence:.2f}/10", (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Arousal: {emotion.arousal:.2f}/10", (25, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Emotion: {emotion.emotion_type.name}", (25, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.0%}", (25, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Thresholds
        cv2.putText(frame, f"Thresh: POS>{config.valence_positive_threshold:.1f} NEG<{config.valence_negative_threshold:.1f}", 
                    (25, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        cv2.imshow("Threshold Testing", frame)
        
        # Log every 30 frames
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"V={emotion.valence:.2f} A={emotion.arousal:.2f} → {emotion.emotion_type.name} ({confidence:.0%})")
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            classifier.reset_smoothing()
            print("Smoothing reset")
        elif key == ord('+') or key == ord('='):
            config.valence_positive_threshold += 0.25
            print(f"Positive threshold: {config.valence_positive_threshold:.2f}")
        elif key == ord('-'):
            config.valence_positive_threshold -= 0.25
            print(f"Positive threshold: {config.valence_positive_threshold:.2f}")
        elif key == ord(']'):
            config.valence_negative_threshold += 0.25
            print(f"Negative threshold: {config.valence_negative_threshold:.2f}")
        elif key == ord('['):
            config.valence_negative_threshold -= 0.25
            print(f"Negative threshold: {config.valence_negative_threshold:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print()
    print("=" * 60)
    print("FINAL THRESHOLDS:")
    print(f"  valence_positive_threshold = {config.valence_positive_threshold}")
    print(f"  valence_negative_threshold = {config.valence_negative_threshold}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test emotion classifier thresholds")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    parser.add_argument("--weights", type=str, default="checkpoints/best_head_ls.pt", help="Model weights path")
    args = parser.parse_args()
    
    test_thresholds_live(camera_id=args.camera, weights_path=args.weights)
