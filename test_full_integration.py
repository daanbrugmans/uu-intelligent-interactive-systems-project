"""Full integration test for emotion-aware cashier conversation system.

Integrates emotion detection, speech recognition, LLM generation, and TTS.
Run full conversation flow with Gemini API and speech I/O.

Usage:
  python test_full_integration.py

Controls:
  SPACE - Start conversation
  r - Reset and start over
  q - Quit
"""

import logging
import subprocess
import threading
import time
from enum import Enum

import cv2
import speech_recognition as sr

from config.settings import Config
from src.conversation_graph import ConversationGraph
from src.emotion_classifier import EmotionClassifier
from src.types import ConversationState

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)



class ConversationMode(Enum):
  """Conversation state machine modes."""

  WAITING_TO_START = "waiting"
  SPEAKING = "speaking"
  LISTENING = "listening"
  PROCESSING = "processing"
  ENDED = "ended"



MICROPHONE_INDEX = 1


class SpeechListener:
  """Manages speech-to-text in background thread using Google API.

  Implements async speech recognition with configurable microphone device
  and noise calibration.
  """
    
  def __init__(self, device_index: int = MICROPHONE_INDEX) -> None:
    """Initializes speech recognition with specified microphone device.

    Args:
      device_index: Microphone device index from system list.
    """
    self.recognizer = sr.Recognizer()
    
    try:
      mics = sr.Microphone.list_microphone_names()
      mic_name = mics[device_index] if device_index < len(mics) else "Unknown"
      logger.info(f"Using microphone: {mic_name}")
      self.microphone = sr.Microphone(device_index=device_index)
    except Exception as e:
      logger.warning(f"Could not use device {device_index}, using default: {e}")
      self.microphone = sr.Microphone()
    
    self._listening = False
    self._result = ""
    self._lock = threading.Lock()
    
    self.recognizer.energy_threshold = 300
    self.recognizer.dynamic_energy_threshold = True
    self.recognizer.pause_threshold = 0.8
    
    with self.microphone as source:
      logger.info("Calibrating for ambient noise (2 seconds)...")
      self.recognizer.adjust_for_ambient_noise(source, duration=2)
      logger.info(f"Energy threshold set to: {self.recognizer.energy_threshold}")
  
  @property
  def is_listening(self) -> bool:
    """Returns whether speech recognition is currently active."""
    with self._lock:
      return self._listening
  
  @property
  def result(self) -> str:
    """Returns the most recent speech recognition result."""
    with self._lock:
      return self._result
  
  def listen_async(
      self, timeout: float = 8.0, callback=None
  ) -> None:
    """Starts speech recognition in background thread.

    Args:
      timeout: Maximum seconds to wait for speech.
      callback: Optional function to call when listening completes.
    """
    def _listen():
      with self._lock:
        self._listening = True
        self._result = ""
      
      try:
        with self.microphone as source:
          logger.info("Listening... (speak now!)")
          audio = self.recognizer.listen(
              source, timeout=timeout, phrase_time_limit=8
          )
        
        try:
          text = self.recognizer.recognize_google(audio, language="en-US")
          with self._lock:
            self._result = text
          logger.info(f'Heard: "{text}"')
          
          text_lower = text.lower()
          if "cash" in text_lower:
            logger.info("Detected: CASH")
          elif "card" in text_lower:
            logger.info("Detected: CARD")
            
        except sr.UnknownValueError:
          logger.info("Could not understand speech - speak louder/clearer")
        except sr.RequestError as e:
          logger.error(f"Speech recognition API error: {e}")
          
      except sr.WaitTimeoutError:
        logger.info("No speech detected (timeout) - speak louder next time")
      except Exception as e:
        logger.error(f"Microphone error: {e}")
      finally:
        with self._lock:
          self._listening = False
        if callback:
          callback()
    
    thread = threading.Thread(target=_listen, daemon=True)
    thread.start()



class Speaker:
  """Thread-safe text-to-speech manager using system say command."""
  
  def __init__(self, voice: str = "Samantha", rate: int = 175) -> None:
    """Initializes speaker with voice and speech rate.

    Args:
      voice: Voice name for TTS (e.g., "Samantha").
      rate: Words per minute for speech rate.
    """
    self.voice = voice
    self.rate = rate
    self._speaking = False
    self._lock = threading.Lock()
  
  @property
  def is_speaking(self) -> bool:
    """Returns whether TTS is currently speaking."""
    with self._lock:
      return self._speaking
  
  def speak(self, text: str, callback=None) -> None:
    """Speaks text asynchronously in background thread.

    Args:
      text: Text to speak.
      callback: Optional function to call when speaking completes.
    """
    def _speak():
      with self._lock:
        self._speaking = True
      try:
        subprocess.run(
            ["say", "-v", self.voice, "-r", str(self.rate), text],
            check=True,
            capture_output=True,
        )
      except Exception as e:
        logger.error(f"TTS error: {e}")
      finally:
        with self._lock:
          self._speaking = False
        if callback:
          callback()
    
    thread = threading.Thread(target=_speak, daemon=True)
    thread.start()
  
  def speak_blocking(self, text: str) -> None:
    """Speaks text synchronously and waits for completion.

    Args:
      text: Text to speak.
    """
    try:
      subprocess.run(
          ["say", "-v", self.voice, "-r", str(self.rate), text],
          check=True,
          capture_output=True,
      )
    except Exception as e:
      logger.error(f"TTS error: {e}")



def draw_ui(
    frame,
    mode: ConversationMode,
    emotion,
    confidence: float,
    response: str,
    round_num: int,
    stage: str,
    countdown: int = 0,
    customer_speech: str = "",
    total: float = 0.0,
    payment: str = "",
):
  """Renders conversation state overlay on video frame.

  Args:
    frame: Video frame to draw on.
    mode: Current conversation mode.
    emotion: Detected emotion object.
    confidence: Confidence percentage.
    response: Agent response text.
    round_num: Current conversation round.
    stage: Conversation stage name.
    countdown: Countdown display value.
    customer_speech: Customer input text.
    total: Transaction total in SEK.
    payment: Payment method name.

  Returns:
    Frame with overlay rendered.
  """
  h, w = frame.shape[:2]
  
  colors = {
      "POSITIVE": (0, 200, 0),
      "NEUTRAL": (200, 200, 0),
      "NEGATIVE": (0, 100, 200),
  }
  mode_colors = {
      ConversationMode.WAITING_TO_START: (100, 100, 100),
      ConversationMode.SPEAKING: (0, 200, 0),
      ConversationMode.LISTENING: (0, 200, 200),
      ConversationMode.PROCESSING: (200, 100, 0),
      ConversationMode.ENDED: (100, 100, 100),
  }
  
  emotion_color = (
      colors.get(emotion.emotion_type.name, (200, 200, 200))
      if emotion
      else (100, 100, 100)
  )
  mode_color = mode_colors.get(mode, (100, 100, 100))
  
  cv2.rectangle(frame, (10, 10), (w - 10, 160), (20, 20, 20), -1)
  cv2.rectangle(frame, (10, 10), (w - 10, 160), emotion_color, 2)
  
  if mode == ConversationMode.LISTENING:
    mode_display = "Speak now..."
  else:
    mode_display = {
        ConversationMode.WAITING_TO_START: "Press SPACE to start",
        ConversationMode.SPEAKING: "AI Speaking...",
        ConversationMode.PROCESSING: "AI Thinking...",
        ConversationMode.ENDED: "Thank you! Goodbye!",
    }.get(mode, "")
  cv2.putText(
      frame, mode_display, (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2
  )
  
  if emotion:
    cv2.putText(
        frame,
        f"Emotion: {emotion.emotion_type.name} ({confidence:.0%})",
        (25, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        emotion_color,
        2,
    )
  
  stage_text = f"Stage: {stage}"
  if total > 0:
    stage_text += f"  |  Total: {total:.2f} SEK"
  if payment:
    stage_text += f"  |  Paying: {payment.upper()}"
  cv2.putText(
      frame, stage_text, (25, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1
  )
  
  if customer_speech:
    speech_display = (
        customer_speech
        if len(customer_speech) <= 50
        else customer_speech[:50] + "..."
    )
    cv2.putText(
        frame,
        f'You said: "{speech_display}"',
        (25, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 200, 200),
        1,
    )
  
  if response:
    cv2.rectangle(frame, (10, h - 80), (w - 10, h - 10), (20, 20, 20), -1)
    cv2.rectangle(frame, (10, h - 80), (w - 10, h - 10), (0, 200, 0), 2)
    
    max_chars = 60
    display_response = (
        response if len(response) <= max_chars else response[:max_chars] + "..."
    )
    cv2.putText(
        frame,
        f'"{display_response}"',
        (25, h - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
  
  return frame



def main() -> None:
  """Runs complete emotion-aware cashier conversation system.

  Coordinates emotion detection, speech recognition, LLM generation, and TTS.
  Implements state machine for conversation flow with payment handling.
  """
  print("\n" + "=" * 70)
  print("  EMOTION-AWARE CASHIER CONVERSATION SYSTEM")
  print("  Webcam → ConvNeXt → LangGraph + Gemini → Speech")
  print("=" * 70 + "\n")
    
  try:
    Config.validate()
    print(f"API Key: {Config.GEMINI_API_KEY[:20]}...{Config.GEMINI_API_KEY[-4:]}")
  except ValueError as e:
    print(f"Configuration error: {e}")
    return
  
  print("\nLoading emotion classifier...")
  classifier = EmotionClassifier()
  
  print("Initializing conversation graph...")
  graph = ConversationGraph(Config.GEMINI_API_KEY)
  
  print("Initializing text-to-speech...")
  speaker = Speaker(voice="Samantha", rate=175)
  
  print("Initializing speech recognition...")
  listener = SpeechListener()
  
  print("Opening webcam...")
  cap = cv2.VideoCapture(0)
  if not cap.isOpened():
    print("Cannot open camera")
    return
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
  
  print("\n" + "=" * 70)
  print("CONTROLS:")
  print("  SPACE - Start conversation")
  print("  r     - Reset conversation")
  print("  q     - Quit")
  print("=" * 70 + "\n")
    
  mode = ConversationMode.WAITING_TO_START
  session = ConversationState(session_id="conversation-001")
  current_emotion = None
  current_confidence = 0.0
  last_response = ""
  current_stage = "waiting"
  customer_speech = ""
  
  MAX_ROUNDS = 10
  last_turn_end_time = 0
  listening_countdown = 0
  started_listening = False
    
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    
    emotion, confidence = classifier.predict(frame)
    current_emotion = emotion
    current_confidence = confidence
    
    if mode != ConversationMode.WAITING_TO_START:
      round_num = session.conversation_round
      if session.is_complete:
        current_stage = "complete"
      elif session.payment_method:
        current_stage = "closing"
      elif session.asked_payment:
        current_stage = "payment_processing"
      elif round_num == 0:
        current_stage = "introduction"
      elif round_num == 1:
        current_stage = "engagement"
      elif round_num == 2:
        current_stage = "weather"
      elif round_num == 3:
        current_stage = "discounts"
      elif round_num == 4:
        current_stage = "assistance"
      elif round_num == 5:
        current_stage = "transition_to_payment"
      elif round_num == 6:
        current_stage = "payment"
      elif round_num == 7:
        current_stage = "payment_processing"
      elif round_num == 8:
        current_stage = "closing"
      else:
        current_stage = "farewell"
        
    if mode == ConversationMode.WAITING_TO_START:
      pass
    
    elif mode == ConversationMode.SPEAKING:
      if not speaker.is_speaking:
        last_turn_end_time = time.time()
        
        if session.is_complete:
          mode = ConversationMode.ENDED
          print("\nConversation completed! Thank you for shopping!")
        elif session.payment_method and session.asked_payment:
          last_response = session.agent_response.lower()
          if any(
              x in last_response
              for x in ["receipt", "change", "thank you"]
          ):
            mode = ConversationMode.ENDED
            print("\nTransaction complete! Goodbye!")
          else:
            print("\nPayment confirmed - proceeding to receipt...")
            mode = ConversationMode.PROCESSING
            customer_speech = ""
        elif len(session.conversation_history) >= MAX_ROUNDS:
          mode = ConversationMode.ENDED
          print("\nConversation completed! (max rounds reached)")
        else:
          mode = ConversationMode.LISTENING
          started_listening = False
          print("\nListening for your response...")
    
    elif mode == ConversationMode.LISTENING:
      if not started_listening:
        started_listening = True
        listener.listen_async(timeout=8.0)
      
      if not listener.is_listening:
        customer_speech = listener.result
        if customer_speech:
          print(f'You said: "{customer_speech}"')
        else:
          customer_speech = ""
        mode = ConversationMode.PROCESSING
    
    elif mode == ConversationMode.PROCESSING:
      print(f"\n--- Round {session.conversation_round + 1} ---")
      print(f"Emotion: {emotion.emotion_type.name} ({confidence:.0%})")
      if customer_speech:
        print(f'Customer said: "{customer_speech}"')
      
      start_time = time.time()
      result = graph.invoke({
          "state": session,
          "emotion": emotion,
          "confidence": confidence,
          "continue_conversation": True,
          "speech_duration": 0.0,
          "customer_speech": customer_speech,
      })
      api_time = time.time() - start_time
      
      session = result["state"]
      last_response = session.agent_response
      customer_speech = ""
      
      print(f'Response: "{last_response}"')
      print(f"Generated in {api_time:.2f}s")
      
      if session.payment_method and any(
          x in last_response.lower() for x in ["receipt", "thank"]
      ):
        session.is_complete = True
        print("Transaction complete - ending after this message")
      
      if len(session.conversation_history) >= 4 and session.payment_method:
        session.is_complete = True
      
      mode = ConversationMode.SPEAKING
      speaker.speak(last_response)
    
    elif mode == ConversationMode.ENDED:
      pass
        
    frame = draw_ui(
        frame,
        mode,
        current_emotion,
        current_confidence,
        last_response,
        session.conversation_round,
        current_stage,
        countdown=listening_countdown,
        customer_speech=session.customer_speech,
        total=session.total_amount,
        payment=session.payment_method or "",
    )
    
    cv2.imshow("Emotion-Aware AI Cashier", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
      print("\nQuitting...")
      break
    
    elif key == ord(' ') and mode == ConversationMode.WAITING_TO_START:
      print("\nStarting conversation...")
      mode = ConversationMode.PROCESSING
    
    elif key == ord('r'):
      print("\nResetting conversation...")
      session = ConversationState(session_id=f"conversation-{int(time.time())}")
      classifier.reset_smoothing()
      last_response = ""
      mode = ConversationMode.WAITING_TO_START
      current_stage = "waiting"
    
  cap.release()
  cv2.destroyAllWindows()
  
  print("\n" + "=" * 70)
  print("SESSION SUMMARY")
  print("=" * 70)
  print(f"  Rounds completed: {session.conversation_round}")
  print(f"  Duration: {session.elapsed_seconds():.1f}s")
  
  if session.emotion_history:
    emotion_counts = {}
    for e in session.emotion_history:
      name = e.emotion_type.name
      emotion_counts[name] = emotion_counts.get(name, 0) + 1
    print(f"  Emotion distribution: {emotion_counts}")
  
  if session.conversation_history:
    print("\n  Conversation transcript:")
    for i, turn in enumerate(session.conversation_history):
      print(f'    {i + 1}. [{turn.emotion.emotion_type.name}] "{turn.agent_response}"')
  
  print("=" * 70 + "\n")



if __name__ == "__main__":
  main()
