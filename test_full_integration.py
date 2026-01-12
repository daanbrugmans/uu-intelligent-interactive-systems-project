"""
Full Integration: Emotion-Aware Cashier with Speech Recognition
================================================================

This script runs the complete emotion-aware conversation system:
    1. Captures frames from webcam continuously
    2. Detects emotion using ConvNeXt classifier
    3. LISTENS to customer speech (speech-to-text)
    4. Generates natural responses via LangGraph + Gemini
    5. Speaks responses using text-to-speech
    6. Handles payment flow (cash/card, receipt, change)

Flow:
    [SPACE to start] → Intro → Chat → Payment (cash/card?) → Receipt → Bye

Controls:
    SPACE - Start the conversation (required once)
    q - Quit
    r - Reset and start over
"""

import cv2
import time
import logging
import subprocess
import threading
import speech_recognition as sr
from enum import Enum
from src.emotion_classifier import EmotionClassifier, ClassifierConfig
from src.conversation_graph import ConversationGraph, AgentState
from src.types import ConversationState
from config.settings import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONVERSATION STATES
# ============================================================================

class ConversationMode(Enum):
    WAITING_TO_START = "waiting"    # Waiting for user to press START
    SPEAKING = "speaking"           # TTS is speaking
    LISTENING = "listening"         # Listening to customer speech
    PROCESSING = "processing"       # Generating response
    ENDED = "ended"                 # Conversation finished


# ============================================================================
# SPEECH RECOGNITION
# ============================================================================

# Microphone device index - MacBook Air Microphone
MICROPHONE_INDEX = 1  # Change this if needed (run list_microphone_names() to see options)

class SpeechListener:
    """Speech-to-text listener using Google Speech Recognition."""
    
    def __init__(self, device_index: int = MICROPHONE_INDEX):
        self.recognizer = sr.Recognizer()
        
        # Use specific microphone (MacBook Air Microphone)
        try:
            mics = sr.Microphone.list_microphone_names()
            mic_name = mics[device_index] if device_index < len(mics) else "Unknown"
            logger.info(f"🎤 Using microphone: {mic_name}")
            self.microphone = sr.Microphone(device_index=device_index)
        except Exception as e:
            logger.warning(f"⚠️ Could not use device {device_index}, using default: {e}")
            self.microphone = sr.Microphone()
        
        self._listening = False
        self._result = ""
        self._lock = threading.Lock()
        
        # Improve recognition settings
        self.recognizer.energy_threshold = 300  # Lower = more sensitive
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8  # Shorter pause before end of phrase
        
        # Adjust for ambient noise on startup
        with self.microphone as source:
            logger.info("🎤 Calibrating for ambient noise (2 seconds)...")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            logger.info(f"🎤 Energy threshold set to: {self.recognizer.energy_threshold}")
    
    @property
    def is_listening(self) -> bool:
        with self._lock:
            return self._listening
    
    @property
    def result(self) -> str:
        with self._lock:
            return self._result
    
    def listen_async(self, timeout: float = 8.0, callback=None) -> None:
        """Listen for speech in background thread."""
        def _listen():
            with self._lock:
                self._listening = True
                self._result = ""
            
            try:
                with self.microphone as source:
                    logger.info("🎤 Listening... (speak now!)")
                    audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=8)
                
                # Try to recognize speech
                try:
                    text = self.recognizer.recognize_google(audio, language="en-US")
                    with self._lock:
                        self._result = text
                    logger.info(f"🗣️ Heard: \"{text}\"")
                    
                    # Also check for cash/card specifically
                    text_lower = text.lower()
                    if "cash" in text_lower:
                        logger.info("💵 Detected: CASH")
                    elif "card" in text_lower:
                        logger.info("💳 Detected: CARD")
                        
                except sr.UnknownValueError:
                    logger.info("🔇 Couldn't understand speech - try speaking louder/clearer")
                except sr.RequestError as e:
                    logger.error(f"❌ Speech recognition API error: {e}")
                    
            except sr.WaitTimeoutError:
                logger.info("⏱️ No speech detected (timeout) - speak louder next time!")
            except Exception as e:
                logger.error(f"❌ Microphone error: {e}")
            finally:
                with self._lock:
                    self._listening = False
                if callback:
                    callback()
        
        thread = threading.Thread(target=_listen, daemon=True)
        thread.start()


# ============================================================================
# TEXT-TO-SPEECH
# ============================================================================

class Speaker:
    """Thread-safe text-to-speech manager."""
    
    def __init__(self, voice: str = "Samantha", rate: int = 175):
        self.voice = voice
        self.rate = rate
        self._speaking = False
        self._lock = threading.Lock()
    
    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return self._speaking
    
    def speak(self, text: str, callback=None) -> None:
        """Speak text in background thread."""
        def _speak():
            with self._lock:
                self._speaking = True
            try:
                subprocess.run(
                    ["say", "-v", self.voice, "-r", str(self.rate), text],
                    check=True,
                    capture_output=True
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
        """Speak and wait for completion."""
        try:
            subprocess.run(
                ["say", "-v", self.voice, "-r", str(self.rate), text],
                check=True,
                capture_output=True
            )
        except Exception as e:
            logger.error(f"TTS error: {e}")


# ============================================================================
# UI DRAWING
# ============================================================================

def draw_ui(frame, mode: ConversationMode, emotion, confidence, 
            response, round_num, stage, countdown=0, customer_speech="", total=0.0, payment=""):
    """Draw the conversation UI overlay."""
    h, w = frame.shape[:2]
    
    # Colors
    colors = {
        "POSITIVE": (0, 200, 0),
        "NEUTRAL": (200, 200, 0),
        "NEGATIVE": (0, 100, 200),
    }
    mode_colors = {
        ConversationMode.WAITING_TO_START: (100, 100, 100),
        ConversationMode.SPEAKING: (0, 200, 0),
        ConversationMode.LISTENING: (0, 200, 200),  # Cyan for listening
        ConversationMode.PROCESSING: (200, 100, 0),
        ConversationMode.ENDED: (100, 100, 100),
    }
    
    emotion_color = colors.get(emotion.emotion_type.name, (200, 200, 200)) if emotion else (100, 100, 100)
    mode_color = mode_colors.get(mode, (100, 100, 100))
    
    # Main info box (top)
    cv2.rectangle(frame, (10, 10), (w - 10, 160), (20, 20, 20), -1)
    cv2.rectangle(frame, (10, 10), (w - 10, 160), emotion_color, 2)
    
    # Mode indicator
    if mode == ConversationMode.LISTENING:
        mode_display = "🎤 Speak now..."
    else:
        mode_display = {
            ConversationMode.WAITING_TO_START: "Press SPACE to start",
            ConversationMode.SPEAKING: "AI Speaking...",
            ConversationMode.PROCESSING: "AI Thinking...",
            ConversationMode.ENDED: "Thank you! Goodbye!",
        }.get(mode, "")
    cv2.putText(frame, mode_display, (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
    
    # Emotion and stage info
    if emotion:
        cv2.putText(frame, f"Emotion: {emotion.emotion_type.name} ({confidence:.0%})", 
                    (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_color, 2)
    
    # Stage and payment info
    stage_text = f"Stage: {stage}"
    if total > 0:
        stage_text += f"  |  Total: {total:.2f} SEK"
    if payment:
        stage_text += f"  |  Paying: {payment.upper()}"
    cv2.putText(frame, stage_text, (25, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    # Customer speech if available
    if customer_speech:
        speech_display = customer_speech if len(customer_speech) <= 50 else customer_speech[:50] + "..."
        cv2.putText(frame, f'You said: "{speech_display}"', (25, 145), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)
    
    # Response box (bottom)
    if response:
        cv2.rectangle(frame, (10, h - 80), (w - 10, h - 10), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, h - 80), (w - 10, h - 10), (0, 200, 0), 2)
        
        # Wrap text if too long
        max_chars = 60
        display_response = response if len(response) <= max_chars else response[:max_chars] + "..."
        cv2.putText(frame, f'"{display_response}"', 
                    (25, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame


# ============================================================================
# MAIN CONVERSATION LOOP
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("  EMOTION-AWARE CASHIER CONVERSATION SYSTEM")
    print("  Webcam → ConvNeXt → LangGraph + Gemini → Speech")
    print("=" * 70 + "\n")
    
    # ──────────────────────────────────────────────────────────────────────
    # INITIALIZATION
    # ──────────────────────────────────────────────────────────────────────
    
    # Validate API key
    try:
        Config.validate()
        print(f"✅ API Key: {Config.GEMINI_API_KEY[:20]}...{Config.GEMINI_API_KEY[-4:]}")
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # Initialize components
    print("\n📊 Loading emotion classifier...")
    classifier = EmotionClassifier()
    
    print("🤖 Initializing conversation graph...")
    graph = ConversationGraph(Config.GEMINI_API_KEY)
    
    print("🔊 Initializing text-to-speech...")
    speaker = Speaker(voice="Samantha", rate=175)
    
    print("🎤 Initializing speech recognition...")
    listener = SpeechListener()
    
    # Open webcam
    print("📷 Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n" + "=" * 70)
    print("CONTROLS:")
    print("  SPACE - Start conversation")
    print("  r     - Reset conversation")
    print("  q     - Quit")
    print("=" * 70 + "\n")
    
    # ──────────────────────────────────────────────────────────────────────
    # STATE
    # ──────────────────────────────────────────────────────────────────────
    
    mode = ConversationMode.WAITING_TO_START
    session = ConversationState(session_id="conversation-001")
    current_emotion = None
    current_confidence = 0.0
    last_response = ""
    current_stage = "waiting"
    customer_speech = ""  # What the customer said
    
    # Timing
    MAX_ROUNDS = 6  # End conversation after this many rounds
    last_turn_end_time = 0
    listening_countdown = 0  # For display
    started_listening = False  # Track if we started listening
    
    # ──────────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ──────────────────────────────────────────────────────────────────────
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Always detect emotion
        emotion, confidence = classifier.predict(frame)
        current_emotion = emotion
        current_confidence = confidence
        
        # Get current stage name (payment-aware)
        if mode != ConversationMode.WAITING_TO_START:
            round_num = session.conversation_round
            # Check if complete first
            if session.is_complete:
                current_stage = "complete"
            # Payment flow takes priority
            elif session.payment_method:
                current_stage = "closing"
            elif session.asked_payment:
                current_stage = "payment_processing"
            elif round_num == 0:
                current_stage = "introduction"
            elif round_num == 1:
                current_stage = "engagement"
            elif round_num == 2:
                current_stage = "assistance"
            elif round_num == 3:
                current_stage = "payment"
            else:
                current_stage = "farewell"
        
        # ──────────────────────────────────────────────────────────────────
        # STATE MACHINE
        # ──────────────────────────────────────────────────────────────────
        
        if mode == ConversationMode.WAITING_TO_START:
            # Just display, wait for SPACE
            pass
        
        elif mode == ConversationMode.SPEAKING:
            # Wait for TTS to finish
            if not speaker.is_speaking:
                last_turn_end_time = time.time()
                
                # Check if conversation should end
                if session.is_complete:
                    mode = ConversationMode.ENDED
                    print("\n🏁 Conversation completed! Thank you for shopping!")
                elif len(session.conversation_history) >= MAX_ROUNDS:
                    mode = ConversationMode.ENDED
                    print("\n🏁 Conversation completed!")
                elif session.payment_method and session.asked_payment:
                    # Payment method was just confirmed - go directly to closing (no listen needed)
                    # Check if we already gave the receipt (contains "receipt" or "change")
                    last_response = session.agent_response.lower()
                    if "receipt" in last_response or "change" in last_response or "thank you" in last_response:
                        mode = ConversationMode.ENDED
                        print("\n🏁 Transaction complete! Goodbye!")
                    else:
                        # Need one more round to give receipt
                        print("\n⏭️  Payment confirmed - proceeding to receipt...")
                        mode = ConversationMode.PROCESSING
                        customer_speech = ""  # No speech needed
                else:
                    mode = ConversationMode.LISTENING
                    started_listening = False
                    print(f"\n🎤 Listening for your response...")
        
        elif mode == ConversationMode.LISTENING:
            # Start speech recognition if not already started
            if not started_listening:
                started_listening = True
                listener.listen_async(timeout=8.0)
            
            # Wait for speech recognition to complete
            if not listener.is_listening:
                customer_speech = listener.result
                if customer_speech:
                    print(f"🗣️ You said: \"{customer_speech}\"")
                else:
                    customer_speech = ""  # No speech detected
                mode = ConversationMode.PROCESSING
        
        elif mode == ConversationMode.PROCESSING:
            # Generate next response
            print(f"\n--- Round {session.conversation_round + 1} ---")
            print(f"📷 Emotion: {emotion.emotion_type.name} ({confidence:.0%})")
            if customer_speech:
                print(f"🗣️ Customer said: \"{customer_speech}\"")
            
            start_time = time.time()
            result = graph.invoke({
                "state": session,
                "emotion": emotion,
                "confidence": confidence,
                "continue_conversation": True,
                "speech_duration": 0.0,
                "customer_speech": customer_speech  # Pass what customer said
            })
            api_time = time.time() - start_time
            
            session = result["state"]
            last_response = session.agent_response
            customer_speech = ""  # Reset for next round
            
            print(f"💬 Response: \"{last_response}\"")
            print(f"⏱️  Generated in {api_time:.2f}s")
            
            # Check if this was the closing (receipt given) - END the conversation
            if session.payment_method and ("receipt" in last_response.lower() or "thank" in last_response.lower()):
                session.is_complete = True  # Mark as done
                print("✅ Transaction complete - ending after this message")
            
            # Also check if we're past the normal rounds
            if len(session.conversation_history) >= 4 and session.payment_method:
                session.is_complete = True
            
            # Start speaking
            mode = ConversationMode.SPEAKING
            speaker.speak(last_response)
        
        elif mode == ConversationMode.ENDED:
            # Just display end state
            pass
        
        # ──────────────────────────────────────────────────────────────────
        # DRAW UI
        # ──────────────────────────────────────────────────────────────────
        
        frame = draw_ui(
            frame, mode, current_emotion, current_confidence,
            last_response, session.conversation_round, current_stage,
            countdown=listening_countdown,
            customer_speech=session.customer_speech,
            total=session.total_amount,
            payment=session.payment_method or ""
        )
        
        cv2.imshow("Emotion-Aware AI Cashier", frame)
        
        # ──────────────────────────────────────────────────────────────────
        # HANDLE INPUT
        # ──────────────────────────────────────────────────────────────────
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n👋 Quitting...")
            break
        
        elif key == ord(' ') and mode == ConversationMode.WAITING_TO_START:
            print("\n🚀 Starting conversation...")
            mode = ConversationMode.PROCESSING
        
        elif key == ord('r'):
            print("\n🔄 Resetting conversation...")
            session = ConversationState(session_id=f"conversation-{int(time.time())}")
            classifier.reset_smoothing()
            last_response = ""
            mode = ConversationMode.WAITING_TO_START
            current_stage = "waiting"
    
    # ──────────────────────────────────────────────────────────────────────
    # CLEANUP
    # ──────────────────────────────────────────────────────────────────────
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
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
            print(f"    {i+1}. [{turn.emotion.emotion_type.name}] \"{turn.agent_response}\"")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
