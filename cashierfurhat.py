#!/usr/bin/env python3
"""
Furhat Emotion-Aware Cashier
============================
Production implementation using Furhat for speech I/O.

Components:
    - Furhat: Speech output (say) and input (listen)
    - ConvNeXt: Real-time emotion detection from webcam
    - LangGraph + Gemini: Natural conversation generation
    - OpenCV: GUI overlay for monitoring

Usage:
    python cashierfurhat.py                 # localhost
    python cashierfurhat.py 192.168.1.100   # custom IP
"""

import cv2
import time
import sys
from enum import Enum

from furhat_remote_api import FurhatRemoteAPI

from src.emotion_classifier import EmotionClassifier
from src.conversation_graph import ConversationGraph
from src.types import ConversationState, EmotionType
from config.settings import Config


# ============================================================================
# CONFIGURATION
# ============================================================================

FURHAT_CHARACTER = "Isabel"
FURHAT_MASK = "Adult"
FURHAT_VOICE = "Joanna"
MAX_ROUNDS = 10  # Extended to match test_full_integration


# ============================================================================
# FURHAT EMOTION EXPRESSIONS
# ============================================================================

def express_emotion(furhat, emotion_type, context="default"):
    """
    Make Furhat express emotions with appropriate gestures and expressions.
    
    Args:
        furhat: FurhatRemoteAPI instance
        emotion_type: EmotionType (POSITIVE, NEUTRAL, NEGATIVE)
        context: Stage context (introduction, payment, assistance, etc.)
    """
    try:
        if emotion_type == EmotionType.POSITIVE:
            # Happy/satisfied
            furhat.gesture(name="Smile")
            time.sleep(0.3)
        elif emotion_type == EmotionType.NEGATIVE:
            # Empathetic/concerned
            if context == "assistance" or context == "handling_negatives":
                # Listening carefully, showing concern
                furhat.gesture(name="Nod")
                time.sleep(0.2)
            else:
                # General empathetic expression
                furhat.gesture(name="Sad")
                time.sleep(0.3)
        else:  # NEUTRAL
            # Calm and professional
            furhat.gesture(name="Nod")
            time.sleep(0.2)
    except Exception as e:
        pass  # Silently fail if Furhat doesn't support the gesture


def get_stage_context_gesture(stage):
    """Get appropriate Furhat gesture for conversation stage."""
    gestures = {
        "introduction": "BigSmile",     # Big welcoming smile
        "engagement": "Smile",           # Friendly
        "weather": "SmallSmile",         # Light smile
        "discounts": "BrowRaise",        # Excited about deals
        "assistance": "Nod",             # Listening/helpful
        "handling_negatives": "Sad",     # Empathetic
        "handling_negatives_result": "Nod",  # Understanding
        "transition_to_payment": "Smile",    # Moving along naturally
        "payment": "Nod",                # Waiting for answer
        "payment_processing": "Nod",     # Processing
        "closing": "Smile",              # Warm goodbye
    }
    return gestures.get(stage, "Nod")


class Mode(Enum):
    WAITING = "waiting"
    SPEAKING = "speaking"
    LISTENING = "listening"
    PROCESSING = "processing"
    ENDED = "ended"


# ============================================================================
# GUI OVERLAY
# ============================================================================

def draw_overlay(frame, mode, emotion, confidence, response, stage, 
                 customer_speech="", total=0.0, payment=""):
    """Draw status overlay on video frame."""
    h, w = frame.shape[:2]
    
    colors = {
        "POSITIVE": (0, 200, 0),
        "NEUTRAL": (200, 200, 0),
        "NEGATIVE": (0, 100, 200),
    }
    
    e_color = colors.get(emotion.emotion_type.name, (150, 150, 150)) if emotion else (150, 150, 150)
    
    # Top info box
    cv2.rectangle(frame, (10, 10), (w-10, 140), (20, 20, 20), -1)
    cv2.rectangle(frame, (10, 10), (w-10, 140), e_color, 2)
    
    # Mode text
    mode_labels = {
        Mode.WAITING: "Press SPACE to start",
        Mode.SPEAKING: "Furhat Speaking...",
        Mode.LISTENING: "Furhat Listening...",
        Mode.PROCESSING: "Generating response...",
        Mode.ENDED: "Conversation ended",
    }
    cv2.putText(frame, mode_labels.get(mode, ""), (25, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, e_color, 2)
    
    # Emotion
    if emotion:
        cv2.putText(frame, f"Emotion: {emotion.emotion_type.name} ({confidence:.0%})", 
                    (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, e_color, 2)
    
    # Stage + payment
    info = f"Stage: {stage}"
    if total > 0:
        info += f" | {total:.2f} SEK"
    if payment:
        info += f" | {payment.upper()}"
    cv2.putText(frame, info, (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    # Customer speech
    if customer_speech:
        cv2.putText(frame, f'You: "{customer_speech[:50]}"', 
                    (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1)
    
    # Response box (bottom)
    if response:
        cv2.rectangle(frame, (10, h-70), (w-10, h-10), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, h-70), (w-10, h-10), (0, 200, 0), 2)
        txt = response[:70] + "..." if len(response) > 70 else response
        cv2.putText(frame, f'AI: "{txt}"', (25, h-35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    furhat_ip = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    
    print("\n" + "="*60)
    print("  FURHAT EMOTION-AWARE CASHIER")
    print("="*60)
    
    # 1. Validate API key
    try:
        Config.validate()
        print("✅ API key loaded")
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # 2. Connect to Furhat
    print(f"🤖 Connecting to Furhat ({furhat_ip})...")
    try:
        furhat = FurhatRemoteAPI(furhat_ip)
        furhat.set_face(character=FURHAT_CHARACTER, mask=FURHAT_MASK)
        furhat.set_voice(name=FURHAT_VOICE)
        print(f"✅ Furhat ready ({FURHAT_CHARACTER}, {FURHAT_VOICE})")
    except Exception as e:
        print(f"❌ Furhat error: {e}")
        return
    
    # 3. Load emotion classifier
    print("📊 Loading emotion classifier...")
    classifier = EmotionClassifier()
    print("✅ Classifier ready")
    
    # 4. Initialize conversation graph
    print("🧠 Initializing LangGraph...")
    graph = ConversationGraph(Config.GEMINI_API_KEY)
    print("✅ Graph ready")
    
    # 5. Open webcam
    print("📷 Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("✅ Webcam ready")
    
    print("\n" + "="*60)
    print("CONTROLS: SPACE=Start | R=Reset | Q=Quit")
    print("="*60 + "\n")
    
    # State
    mode = Mode.WAITING
    session = ConversationState(session_id=f"session-{int(time.time())}")
    emotion = None
    confidence = 0.0
    response = ""
    speech = ""
    started_listening = False
    
    def get_stage(round_num):
        """Get stage name based on round number - STRICT SEQUENCE."""
        stages_sequence = [
            "introduction",          # Round 0
            "engagement",            # Round 1
            "weather",               # Round 2
            "discounts",             # Round 3
            "assistance",            # Round 4
            "transition_to_payment", # Round 5
            "payment",               # Round 6
            "payment_processing",    # Round 7
            "closing",               # Round 8
            "farewell"               # Round 9
        ]
        
        # Handle payment method confirmation
        if session.payment_method:
            return "closing"
        if session.asked_payment and not session.payment_method:
            return "payment_processing"
        
        # Handle item lookup flow
        if session.looking_up_item:
            if session.item_found is None:
                return "handling_negatives_result"
            else:
                if session.item_found:
                    session.total_amount += 15.0
                session.item_found = None
                session.looking_up_item = False
        
        # Handle "no" at assistance stage
        if round_num == 4 and session.customer_speech:
            speech_lower = session.customer_speech.lower()
            no_keywords = ["no", "didn't", "don't", "nope", "nah", "didn't find", "don't have", "can't find", "couldn't find", "don't see"]
            if any(kw in speech_lower for kw in no_keywords):
                session.looking_up_item = True
                return "handling_negatives"
        
        if session.is_complete:
            return "complete"
        
        # Return stage from sequence
        if round_num < len(stages_sequence):
            return stages_sequence[round_num]
        return "farewell"
    
    # ──────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ──────────────────────────────────────────────────────────────
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Always detect emotion from webcam
        emotion, confidence = classifier.predict(frame)
        stage = get_stage(session.conversation_round) if mode != Mode.WAITING else "waiting"
        
        # ──────────────────────────────────────────────────────────
        # STATE MACHINE
        # ──────────────────────────────────────────────────────────
        
        if mode == Mode.WAITING:
            pass  # Wait for SPACE
        
        elif mode == Mode.PROCESSING:
            print(f"\n--- Round {session.conversation_round + 1} | {stage} ---")
            print(f"📷 {emotion.emotion_type.name} ({confidence:.0%})")
            if speech:
                print(f"🗣️ Customer: \"{speech}\"")
            
            # Generate LLM response
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
            speech = ""  # Clear for next round
            
            print(f"🤖 Response: \"{response}\"")
            
            # Check if conversation should end
            resp_lower = response.lower()
            if session.payment_method and any(x in resp_lower for x in ["receipt", "change", "thank you", "goodbye"]):
                session.is_complete = True
            
            # Get current stage for emotion expression
            current_stage = get_stage(session.conversation_round)
            
            # Furhat expresses emotion
            express_emotion(furhat, emotion.emotion_type, current_stage)
            
            # Try stage-specific gesture, fallback to emotion-based
            try:
                gesture = get_stage_context_gesture(current_stage)
                furhat.gesture(name=gesture)
            except:
                pass
            
            # Furhat speaks
            mode = Mode.SPEAKING
            furhat.say(text=response, blocking=True)
            
            # Decide next state
            if session.is_complete:
                mode = Mode.ENDED
                print("\n🏁 Conversation complete!")
            elif session.payment_method and session.asked_payment:
                # Auto-proceed after payment confirmation
                if "receipt" in resp_lower or "change" in resp_lower:
                    mode = Mode.ENDED
                else:
                    mode = Mode.PROCESSING  # Generate receipt
            elif len(session.conversation_history) >= MAX_ROUNDS:
                mode = Mode.ENDED
                print("\n🏁 Max rounds reached!")
            else:
                mode = Mode.LISTENING
                started_listening = False
        
        elif mode == Mode.LISTENING:
            print("🎤 Furhat listening...")
            
            # Express listening emotion
            express_emotion(furhat, emotion.emotion_type, "assistance")
            furhat.gesture(name="Nod")
            
            # Furhat listens for customer speech
            result = furhat.listen()
            speech = result.message if result and result.message else ""
            
            if speech:
                print(f"🗣️ Heard: \"{speech}\"")
                session.customer_speech = speech  # Store what customer said
            else:
                print("🔇 No speech detected")
            
            mode = Mode.PROCESSING
        
        elif mode == Mode.ENDED:
            pass  # Display end state
        
        # ──────────────────────────────────────────────────────────
        # DRAW GUI
        # ──────────────────────────────────────────────────────────
        
        draw_overlay(
            frame, mode, emotion, confidence, response, stage,
            session.customer_speech, session.total_amount, 
            session.payment_method or ""
        )
        cv2.imshow("Furhat Cashier", frame)
        
        # ──────────────────────────────────────────────────────────
        # KEYBOARD INPUT
        # ──────────────────────────────────────────────────────────
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n👋 Quitting...")
            break
        
        elif key == ord(' ') and mode == Mode.WAITING:
            print("\n🚀 Starting conversation...")
            mode = Mode.PROCESSING
        
        elif key == ord('r'):
            print("\n🔄 Resetting...")
            session = ConversationState(session_id=f"session-{int(time.time())}")
            classifier.reset_smoothing()
            response = ""
            speech = ""
            mode = Mode.WAITING
    
    # ──────────────────────────────────────────────────────────────
    # CLEANUP
    # ──────────────────────────────────────────────────────────────
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print("\n" + "="*60)
    print("SESSION SUMMARY")
    print(f"  Rounds: {session.conversation_round}")
    print(f"  Duration: {session.elapsed_seconds():.1f}s")
    if session.payment_method:
        print(f"  Payment: {session.payment_method.upper()}")
    if session.conversation_history:
        print("\n  Transcript:")
        for i, turn in enumerate(session.conversation_history):
            e = turn.emotion.emotion_type.name if turn.emotion else "?"
            txt = turn.agent_response[:55] + "..." if len(turn.agent_response) > 55 else turn.agent_response
            print(f"    {i+1}. [{e}] {txt}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
