#!/usr/bin/env python3
"""Production cashier system with Furhat, emotion detection, and LLM.

Integrates:
- Furhat robot for speech I/O and gestures
- ConvNeXt for real-time emotion detection
- LangGraph + Gemini for conversation generation
- OpenCV for monitoring and UI overlay

Usage:
  python cashierfurhat.py                # localhost
  python cashierfurhat.py 192.168.1.100  # custom IP
"""

import sys
import time
from enum import Enum

import cv2
import numpy as np
from furhat_remote_api import FurhatRemoteAPI

from config.settings import Config
from src.conversation_graph import ConversationGraph
from src.emotion_classifier import EmotionClassifier
from src.types import ConversationState, Emotion, EmotionType

FURHAT_CHARACTER = "Isabel"
FURHAT_MASK = "Adult"
FURHAT_VOICE = "Joanna"
MAX_ROUNDS = 10


def express_emotion(
    furhat: "FurhatRemoteAPI", emotion_type: EmotionType, context: str = "default"
) -> None:
  """Makes Furhat express emotions with appropriate gestures.
  
  Args:
    furhat: FurhatRemoteAPI instance.
    emotion_type: Detected emotion type.
    context: Conversation stage context.
  """
  try:
    if emotion_type == EmotionType.POSITIVE:
      furhat.gesture(name="Smile")
      time.sleep(0.3)
    elif emotion_type == EmotionType.NEGATIVE:
      if context in ("assistance", "handling_negatives"):
        furhat.gesture(name="Nod")
        time.sleep(0.2)
      else:
        furhat.gesture(name="Sad")
        time.sleep(0.3)
    else:
      furhat.gesture(name="Nod")
      time.sleep(0.2)
  except Exception:
    pass


def get_stage_context_gesture(stage: str) -> str:
  """Gets appropriate Furhat gesture for conversation stage.
  
  Args:
    stage: Conversation stage name.
  
  Returns:
    Gesture name for Furhat.
  """
  gestures = {
      "introduction": "BigSmile",
      "engagement": "Smile",
      "weather": "SmallSmile",
      "discounts": "BrowRaise",
      "assistance": "Nod",
      "handling_negatives": "Sad",
      "handling_negatives_result": "Nod",
      "transition_to_payment": "Smile",
      "payment": "Nod",
      "payment_processing": "Nod",
      "closing": "Smile",
  }
  return gestures.get(stage, "Nod")


class Mode(Enum):
  """Application state machine modes."""

  WAITING = "waiting"
  SPEAKING = "speaking"
  LISTENING = "listening"
  PROCESSING = "processing"
  ENDED = "ended"


def draw_overlay(
    frame: np.ndarray,
    mode: Mode,
    emotion: Emotion | None,
    confidence: float,
    response: str,
    stage: str,
    customer_speech: str = "",
    total: float = 0.0,
    payment: str = "",
) -> np.ndarray:
  """Renders conversation state overlay on video frame.

  Displays conversation mode, emotion detection, stage, speech transcripts,
  and payment information on the video feed with color-coded borders.

  Args:
    frame: The video frame to draw on.
    mode: Current application mode.
    emotion: Detected emotion object or None.
    confidence: Detection confidence as percentage.
    response: Agent response text to display.
    stage: Current conversation stage.
    customer_speech: Customer speech transcript.
    total: Transaction amount in SEK.
    payment: Selected payment method.

  Returns:
    The frame with overlay rendered.
  """
  h, w = frame.shape[:2]
  
  colors = {
      "POSITIVE": (0, 200, 0),
      "NEUTRAL": (200, 200, 0),
      "NEGATIVE": (0, 100, 200),
  }
  
  emotion_color = (
      colors.get(emotion.emotion_type.name, (150, 150, 150))
      if emotion else (150, 150, 150)
  )
  
  cv2.rectangle(frame, (10, 10), (w - 10, 140), (20, 20, 20), -1)
  cv2.rectangle(frame, (10, 10), (w - 10, 140), emotion_color, 2)
  
  mode_labels = {
      Mode.WAITING: "Press SPACE to start",
      Mode.SPEAKING: "Furhat Speaking...",
      Mode.LISTENING: "Furhat Listening...",
      Mode.PROCESSING: "Generating response...",
      Mode.ENDED: "Conversation ended",
  }
  cv2.putText(
      frame,
      mode_labels.get(mode, ""),
      (25, 40),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.7,
      emotion_color,
      2,
  )
  
  if emotion:
    cv2.putText(
        frame,
        f"Emotion: {emotion.emotion_type.name} ({confidence:.0%})",
        (25, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        emotion_color,
        2,
    )
  
  info = f"Stage: {stage}"
  if total > 0:
    info += f" | {total:.2f} SEK"
  if payment:
    info += f" | {payment.upper()}"
  cv2.putText(
      frame, info, (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1
  )
  
  if customer_speech:
    cv2.putText(
        frame,
        f'You: "{customer_speech[:50]}"',
        (25, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 200, 200),
        1,
    )
  
  if response:
    cv2.rectangle(frame, (10, h - 70), (w - 10, h - 10), (20, 20, 20), -1)
    cv2.rectangle(frame, (10, h - 70), (w - 10, h - 10), (0, 200, 0), 2)
    txt = response[:70] + "..." if len(response) > 70 else response
    cv2.putText(
        frame,
        f'AI: "{txt}"',
        (25, h - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
  
  return frame

def main() -> None:
  """Main application loop coordinating Furhat, emotion detection, and dialogue.

  Initializes Furhat robot, emotion classifier, and conversation graph. Runs
  the state machine event loop coordinating video capture, emotion detection,
  speech recognition, and dialogue generation until user terminates.
  """
  furhat_ip = sys.argv[1] if len(sys.argv) > 1 else "localhost"
  
  print("\n" + "=" * 60)
  print("  FURHAT EMOTION-AWARE CASHIER")
  print("=" * 60)
  
  try:
    Config.validate()
    print("API key loaded")
  except ValueError as e:
    print(f"Configuration error: {e}")
    return
  
  print(f"Connecting to Furhat ({furhat_ip})...")
  try:
    furhat = FurhatRemoteAPI(furhat_ip)
    furhat.set_face(character=FURHAT_CHARACTER, mask=FURHAT_MASK)
    furhat.set_voice(name=FURHAT_VOICE)
    print(f"Furhat ready ({FURHAT_CHARACTER}, {FURHAT_VOICE})")
  except Exception as e:
    print(f"Furhat connection error: {e}")
    return
  
  print("Loading emotion classifier...")
  classifier = EmotionClassifier()
  print("Classifier ready")
  
  print("Initializing conversation graph...")
  graph = ConversationGraph(Config.GEMINI_API_KEY)
  print("Graph ready")
  
  print("Opening webcam...")
  cap = cv2.VideoCapture(0)
  if not cap.isOpened():
    print("Cannot open camera")
    return
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
  print("Webcam ready")
  
  print("\n" + "=" * 60)
  print("CONTROLS: SPACE=Start | R=Reset | Q=Quit")
  print("=" * 60 + "\n")
  
  mode = Mode.WAITING
  session = ConversationState(session_id=f"session-{int(time.time())}")
  emotion = None
  confidence = 0.0
  response = ""
  speech = ""
  
  def get_stage(round_num: int) -> str:
    """Determines conversation stage based on round and session state.

    Implements strict sequence-based stage progression with special handling
    for payment and item lookup flows.

    Args:
      round_num: Current conversation round number.

    Returns:
      The conversation stage identifier.
    """
    stages_sequence = [
        "introduction",
        "engagement",
        "weather",
        "discounts",
        "assistance",
        "transition_to_payment",
        "payment",
        "payment_processing",
        "closing",
        "farewell",
    ]
    
    if session.payment_method:
      return "closing"
    if session.asked_payment and not session.payment_method:
      return "payment_processing"
    
    if session.looking_up_item:
      if session.item_found is None:
        return "handling_negatives_result"
      else:
        if session.item_found:
          session.total_amount += 15.0
        session.item_found = None
        session.looking_up_item = False
    
    if round_num == 4 and session.customer_speech:
      speech_lower = session.customer_speech.lower()
      no_keywords = [
          "no",
          "didn't",
          "don't",
          "nope",
          "nah",
          "didn't find",
          "don't have",
          "can't find",
          "couldn't find",
          "don't see",
      ]
      if any(kw in speech_lower for kw in no_keywords):
        session.looking_up_item = True
        return "handling_negatives"
    
    if session.is_complete:
      return "complete"
    
    if round_num < len(stages_sequence):
      return stages_sequence[round_num]
    return "farewell"
  
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    
    emotion, confidence = classifier.predict(frame)
    stage = get_stage(session.conversation_round) if mode != Mode.WAITING else "waiting"
    
    if mode == Mode.WAITING:
      pass
    
    elif mode == Mode.PROCESSING:
      print(f"\n--- Round {session.conversation_round + 1} | {stage} ---")
      print(f"{emotion.emotion_type.name} ({confidence:.0%})")
      if speech:
        print(f"Customer: \"{speech}\"")
      
      result = graph.invoke({
          "state": session,
          "emotion": emotion,
          "confidence": confidence,
          "continue_conversation": True,
          "speech_duration": 0.0,
          "customer_speech": speech
      })
      
      session = result["state"]
      response = session.agent_response
      speech = ""
      
      print(f"Response: \"{response}\"")
      
      resp_lower = response.lower()
      if session.payment_method and any(
          x in resp_lower for x in ["receipt", "change", "thank you", "goodbye"]
      ):
        session.is_complete = True
      
      current_stage = get_stage(session.conversation_round)
      
      express_emotion(furhat, emotion.emotion_type, current_stage)
      
      try:
        gesture = get_stage_context_gesture(current_stage)
        furhat.gesture(name=gesture)
      except Exception:
        pass
      
      mode = Mode.SPEAKING
      furhat.say(text=response, blocking=True)
      
      if session.is_complete:
        mode = Mode.ENDED
        print("\nConversation complete!")
      elif session.payment_method and session.asked_payment:
        if "receipt" in resp_lower or "change" in resp_lower:
          mode = Mode.ENDED
        else:
          mode = Mode.PROCESSING
      elif len(session.conversation_history) >= MAX_ROUNDS:
        mode = Mode.ENDED
        print("\nMax rounds reached!")
      else:
        mode = Mode.LISTENING
    
    elif mode == Mode.LISTENING:
      print("Furhat listening...")
      
      express_emotion(furhat, emotion.emotion_type, "assistance")
      furhat.gesture(name="Nod")
      
      result = furhat.listen()
      speech = result.message if result and result.message else ""
      
      if speech:
        print(f"Heard: \"{speech}\"")
        session.customer_speech = speech
      else:
        print("No speech detected")
      
      mode = Mode.PROCESSING
    
    elif mode == Mode.ENDED:
      pass
    
    draw_overlay(
        frame,
        mode,
        emotion,
        confidence,
        response,
        stage,
        session.customer_speech,
        session.total_amount,
        session.payment_method or ""
    )
    cv2.imshow("Furhat Cashier", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
      print("\nQuitting...")
      break
    
    elif key == ord(' ') and mode == Mode.WAITING:
      print("\nStarting conversation...")
      mode = Mode.PROCESSING
    
    elif key == ord('r'):
      print("\nResetting...")
      session = ConversationState(session_id=f"session-{int(time.time())}")
      classifier.reset_smoothing()
      response = ""
      speech = ""
      mode = Mode.WAITING
  
  cap.release()
  cv2.destroyAllWindows()
  
  print("\n" + "=" * 60)
  print("SESSION SUMMARY")
  print(f"  Rounds: {session.conversation_round}")
  print(f"  Duration: {session.elapsed_seconds():.1f}s")
  if session.payment_method:
    print(f"  Payment: {session.payment_method.upper()}")
  if session.conversation_history:
    print("\n  Transcript:")
    for i, turn in enumerate(session.conversation_history):
      emotion_name = turn.emotion.emotion_type.name if turn.emotion else "?"
      txt = (
          turn.agent_response[:55] + "..."
          if len(turn.agent_response) > 55
          else turn.agent_response
      )
      print(f"    {i + 1}. [{emotion_name}] {txt}")
  print("=" * 60 + "\n")


if __name__ == "__main__":
  main()
