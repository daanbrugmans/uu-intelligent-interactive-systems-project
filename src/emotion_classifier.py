"""Real-time emotion classification from video frames using ConvNeXt-Tiny.

This module provides emotion classification (POSITIVE, NEUTRAL, NEGATIVE) from
video frames by predicting valence and arousal values using a fine-tuned
ConvNeXt-Tiny model trained on the EMOTIC dataset.
"""

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision

from src.types import Emotion, EmotionType

logger = logging.getLogger(__name__)


@dataclass
class ClassifierConfig:
  """Configuration for the emotion classifier."""

  weights_path: str = "checkpoints/best_head_ls.pt"
  input_size: int = 224
  valence_positive_threshold: float = 7.1
  valence_negative_threshold: float = 6.9
  confidence_scale: float = 2.0
  smoothing_enabled: bool = True
  smoothing_alpha: float = 0.3
  inference_interval_ms: int = 100


class EmotionClassifier:
  """Real-time emotion classifier using ConvNeXt-Tiny.
  
  Predicts valence and arousal from video frames and maps to emotion categories.
  Supports exponential moving average smoothing and frame-rate limiting.
  """

  def __init__(self, config: Optional[ClassifierConfig] = None) -> None:
    """Initializes the classifier.
    
    Args:
      config: Optional configuration. Defaults to ClassifierConfig() if not provided.
    
    Raises:
      FileNotFoundError: If model weights file does not exist.
    """
    self.config = config or ClassifierConfig()
    self.device = self._get_device()
    self.model = self._load_model()

    self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    self.std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    self._smoothed_valence: Optional[float] = None
    self._smoothed_arousal: Optional[float] = None
    self._last_inference_time: float = 0.0
    self._last_result: Optional[Tuple[Emotion, float]] = None

    logger.info(f"EmotionClassifier initialized on {self.device}")

  def _get_device(self) -> torch.device:
    """Returns the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
      return torch.device("mps")
    if torch.cuda.is_available():
      return torch.device("cuda")
    return torch.device("cpu")

  def _load_model(self) -> torch.nn.Module:
    """Loads the fine-tuned ConvNeXt-Tiny model from checkpoint.
    
    Returns:
      The loaded model in evaluation mode on the target device.
    
    Raises:
      FileNotFoundError: If checkpoint path does not exist.
    """
    weights_path = Path(self.config.weights_path)

    if not weights_path.exists():
      raise FileNotFoundError(
          f"Model weights not found at {weights_path}"
      )

    logger.info(f"Loading model from {weights_path}")

    model = torchvision.models.convnext_tiny(weights=None)

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)

    features_state = {
        k.replace("features.", ""): v
        for k, v in state.items()
        if k.startswith("features.")
    }
    model.features.load_state_dict(features_state)

    model.classifier = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.LayerNorm(768, eps=1e-6),
        torch.nn.Linear(768, 512),
        torch.nn.GELU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(512, 2),
    )

    classifier_state = {
        k.replace("classifier.", ""): v
        for k, v in state.items()
        if k.startswith("classifier.")
    }
    model.classifier.load_state_dict(classifier_state)

    model.to(self.device)
    model.eval()

    return model

  def preprocess(self, frame: np.ndarray) -> torch.Tensor:
    """Preprocesses a video frame for model inference.
    
    Args:
      frame: BGR image from OpenCV with shape (H, W, 3).
    
    Returns:
      Preprocessed tensor with shape (1, 3, 224, 224) on the target device.
    """
    frame_resized = cv2.resize(
        frame,
        (self.config.input_size, self.config.input_size),
        interpolation=cv2.INTER_CUBIC,
    )

    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
    tensor = (tensor - self.mean) / self.std

    return tensor.unsqueeze(0).to(self.device)

  def predict(self, frame: np.ndarray) -> Tuple[Emotion, float]:
    """Predicts emotion from a video frame.
    
    Args:
      frame: BGR image from OpenCV.
    
    Returns:
      Tuple of (Emotion object, confidence score in [0, 1]).
    """
    current_time = time.time() * 1000
    if (
        self._last_result is not None
        and current_time - self._last_inference_time
        < self.config.inference_interval_ms
    ):
      return self._last_result

    input_tensor = self.preprocess(frame)

    with torch.no_grad():
      output = self.model(input_tensor)
      pred = output.cpu().numpy()[0]

    raw_valence = float(pred[0] * 9.0 + 1.0)
    raw_arousal = float(pred[1] * 9.0 + 1.0)

    if self.config.smoothing_enabled:
      valence, arousal = self._apply_smoothing(raw_valence, raw_arousal)
    else:
      valence, arousal = raw_valence, raw_arousal

    emotion_type, confidence = self._classify_valence(valence)

    emotion = Emotion(
        emotion_type=emotion_type, valence=valence, arousal=arousal
    )

    self._last_result = (emotion, confidence)
    self._last_inference_time = current_time

    logger.debug(
        f"Predicted: {emotion_type.name} "
        f"(V={valence:.2f}, A={arousal:.2f}, conf={confidence:.0%})"
    )

    return emotion, confidence

  def _apply_smoothing(
      self, valence: float, arousal: float
  ) -> Tuple[float, float]:
    """Applies exponential moving average smoothing to predictions.
    
    Args:
      valence: Raw valence prediction.
      arousal: Raw arousal prediction.
    
    Returns:
      Tuple of (smoothed_valence, smoothed_arousal).
    """
    alpha = self.config.smoothing_alpha

    if self._smoothed_valence is None:
      self._smoothed_valence = valence
      self._smoothed_arousal = arousal
    else:
      self._smoothed_valence = (
          alpha * valence + (1 - alpha) * self._smoothed_valence
      )
      self._smoothed_arousal = (
          alpha * arousal + (1 - alpha) * self._smoothed_arousal
      )

    return self._smoothed_valence, self._smoothed_arousal

  def _classify_valence(
      self, valence: float
  ) -> Tuple[EmotionType, float]:
    """Classifies valence into emotion type and calculates confidence.
    
    Args:
      valence: Valence score in range [1, 10].
    
    Returns:
      Tuple of (EmotionType, confidence in [0, 1]).
    """
    pos_thresh = self.config.valence_positive_threshold
    neg_thresh = self.config.valence_negative_threshold
    mid_point = (pos_thresh + neg_thresh) / 2

    if valence >= pos_thresh:
      emotion_type = EmotionType.POSITIVE
      distance = valence - pos_thresh
      confidence = min(
          1.0, 0.5 + distance * self.config.confidence_scale / 10
      )
    elif valence <= neg_thresh:
      emotion_type = EmotionType.NEGATIVE
      distance = neg_thresh - valence
      confidence = min(
          1.0, 0.5 + distance * self.config.confidence_scale / 10
      )
    else:
      emotion_type = EmotionType.NEUTRAL
      distance_from_center = abs(valence - mid_point)
      max_distance = (pos_thresh - neg_thresh) / 2
      confidence = 0.5 + 0.3 * (1 - distance_from_center / max_distance)

    return emotion_type, confidence

  def reset_smoothing(self) -> None:
    """Resets smoothing state."""
    self._smoothed_valence = None
    self._smoothed_arousal = None
    logger.debug("Smoothing state reset")

  def get_raw_prediction(
      self, frame: np.ndarray
  ) -> Tuple[float, float]:
    """Gets raw valence and arousal without classification.
    
    Args:
      frame: BGR image from OpenCV.
    
    Returns:
      Tuple of (valence in [1, 10], arousal in [1, 10]).
    """
    input_tensor = self.preprocess(frame)

    with torch.no_grad():
      output = self.model(input_tensor)
      pred = output.cpu().numpy()[0]

    valence = float(pred[0] * 9.0 + 1.0)
    arousal = float(pred[1] * 9.0 + 1.0)

    return valence, arousal


def test_thresholds_live(
    camera_id: int = 0, weights_path: str = "checkpoints/best_head_ls.pt"
) -> None:
  """Interactive tool to test and tune emotion classification thresholds.
  
  Displays real-time valence/arousal predictions and emotion classifications
  from a webcam feed.
  
  Args:
    camera_id: Index of the camera device.
    weights_path: Path to model checkpoint file.
  
  Key Controls:
    q: Quit
    r: Reset smoothing state
    +/-: Adjust positive threshold
    [/]: Adjust negative threshold
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
  print("Current thresholds:")
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

  colors = {
      EmotionType.POSITIVE: (0, 255, 0),
      EmotionType.NEUTRAL: (255, 255, 0),
      EmotionType.NEGATIVE: (0, 0, 255),
  }

  frame_count = 0

  while True:
    ret, frame = cap.read()
    if not ret:
      break

    emotion, confidence = classifier.predict(frame)
    color = colors[emotion.emotion_type]

    cv2.rectangle(frame, (10, 10), (400, 200), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (400, 200), color, 3)

    cv2.putText(
        frame,
        f"Valence: {emotion.valence:.2f}/10",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Arousal: {emotion.arousal:.2f}/10",
        (25, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Emotion: {emotion.emotion_type.name}",
        (25, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )
    cv2.putText(
        frame,
        f"Confidence: {confidence:.0%}",
        (25, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Thresh: POS>{config.valence_positive_threshold:.1f} "
        f"NEG<{config.valence_negative_threshold:.1f}",
        (25, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (150, 150, 150),
        1,
    )

    cv2.imshow("Threshold Testing", frame)

    frame_count += 1
    if frame_count % 30 == 0:
      print(
          f"V={emotion.valence:.2f} A={emotion.arousal:.2f} "
          f"→ {emotion.emotion_type.name} ({confidence:.0%})"
      )

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
      break
    elif key == ord("r"):
      classifier.reset_smoothing()
      print("Smoothing reset")
    elif key == ord("+") or key == ord("="):
      config.valence_positive_threshold += 0.25
      print(f"Positive threshold: {config.valence_positive_threshold:.2f}")
    elif key == ord("-"):
      config.valence_positive_threshold -= 0.25
      print(f"Positive threshold: {config.valence_positive_threshold:.2f}")
    elif key == ord("]"):
      config.valence_negative_threshold += 0.25
      print(f"Negative threshold: {config.valence_negative_threshold:.2f}")
    elif key == ord("["):
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
  parser = argparse.ArgumentParser(
      description="Test emotion classifier thresholds"
  )
  parser.add_argument("--camera", type=int, default=0, help="Camera ID")
  parser.add_argument(
      "--weights",
      type=str,
      default="checkpoints/best_head_ls.pt",
      help="Model weights path",
  )
  args = parser.parse_args()

  test_thresholds_live(camera_id=args.camera, weights_path=args.weights)