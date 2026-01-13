"""
Test Script: Visual Emotion Loop - LangGraph + Gemini
======================================================

This test simulates the VISUAL-ONLY emotion loop:
    1. Webcam captures frame (simulated with emotion sequence)
    2. ConvNeXt classifies emotion + confidence
    3. ConversationGraph generates response via Gemini
    4. Response sent to Furhat (simulated with print + wait)
    5. Loop continues

NO text input - purely based on visual emotion recognition.

Expected timing:
    - API call: ~0.5-1.5s
    - Speech duration: ~2-4s per response
    - Total per round: ~3-5s
"""

import logging
import time
from config.settings import Config
from src.conversation_graph import ConversationGraph, AgentState
from src.types import ConversationState, Emotion, EmotionType

# Enable detailed logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def simulate_furhat_speech(text: str, duration: float) -> None:
    """
    Simulate Furhat speaking the response.
    In production, this calls the actual Furhat SDK.
    """
    print(f'  🤖 Furhat: "{text}"')
    print(f"  ⏳ Speaking for {duration:.1f}s...")
    time.sleep(duration)


def test_visual_emotion_loop():
    """
    Test the visual emotion loop with simulated webcam emotions.

    This simulates what happens in the real system:
    - No text input from user
    - Only visual emotion detection
    - Responses based purely on observed emotions
    """

    print("\n" + "=" * 70)
    print("  Visual Emotion Loop Test")
    print("  LangGraph + Gemini + Furhat (simulated)")
    print("=" * 70 + "\n")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 1: Validate configuration
    # ──────────────────────────────────────────────────────────────────────
    try:
        Config.validate()
        print(
            f"✅ API Key: {Config.GEMINI_API_KEY[:20]}...{Config.GEMINI_API_KEY[-4:]}"
        )
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("   Please set GEMINI_API_KEY in your .env file")
        return

    # ──────────────────────────────────────────────────────────────────────
    # STEP 2: Initialize conversation graph
    # ──────────────────────────────────────────────────────────────────────
    print("\n📊 Initializing ConversationGraph (visual mode)...")
    graph = ConversationGraph(Config.GEMINI_API_KEY)
    print("✅ Ready for visual emotion loop\n")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 3: Simulate webcam emotion sequence
    # ──────────────────────────────────────────────────────────────────────
    # Each tuple: (emotion_type, valence, arousal, confidence)
    # This simulates what ConvNeXt would output from webcam frames
    emotion_sequence = [
        # Customer approaches, looks positive/happy
        (EmotionType.POSITIVE, 8.0, 6.0, 0.92),
        # Customer browsing, neutral expression
        (EmotionType.NEUTRAL, 5.0, 5.0, 0.78),
        # Customer looks uncertain/negative about price
        (EmotionType.NEGATIVE, 3.0, 3.0, 0.85),
        # Cashier offers help, customer feels better
        (EmotionType.NEUTRAL, 5.5, 5.0, 0.71),
        # Customer decides to buy, positive again
        (EmotionType.POSITIVE, 8.5, 6.5, 0.89),
    ]

    session = ConversationState(session_id="visual-test-001")
    success_count = 0
    start_time = time.time()

    print("─" * 70)
    print("Starting visual emotion loop (simulated webcam input)")
    print("─" * 70 + "\n")

    for i, (emotion_type, valence, arousal, confidence) in enumerate(emotion_sequence):
        print(f"📷 Frame {i + 1}/{len(emotion_sequence)}")
        print(f"  Detected: {emotion_type.name} (confidence: {confidence:.0%})")
        print(f"  Valence: {valence}, Arousal: {arousal}")

        # Create emotion object (simulating ConvNeXt output)
        emotion = Emotion(emotion_type, valence, arousal)

        # Invoke graph (visual-only - no text input)
        turn_start = time.time()
        result = graph.invoke(
            {
                "state": session,
                "emotion": emotion,
                "confidence": confidence,
                "continue_conversation": True,
                "speech_duration": 0.0,
            }
        )
        api_time = time.time() - turn_start

        # Extract results
        session = result["state"]
        speech_duration = result.get("speech_duration", 2.0)

        # Simulate Furhat speaking
        simulate_furhat_speech(session.agent_response, speech_duration)

        # Stats
        total_turn_time = time.time() - turn_start
        print(f"  📊 API: {api_time:.2f}s | Total: {total_turn_time:.2f}s")
        print()

        if session.agent_response:
            success_count += 1

        # Check if we should continue
        if not result["continue_conversation"]:
            print("🏁 Loop terminated by graph")
            break

    # ──────────────────────────────────────────────────────────────────────
    # STEP 4: Summary
    # ──────────────────────────────────────────────────────────────────────
    total_duration = time.time() - start_time
    print("─" * 70)
    print("  VISUAL LOOP TEST SUMMARY")
    print("─" * 70)
    print(f"  ✅ Completed: {success_count}/{len(emotion_sequence)} rounds")
    print(f"  ⏱️  Total time: {total_duration:.1f}s")
    print(f"  📊 Avg per round: {total_duration / len(emotion_sequence):.1f}s")
    print("─" * 70 + "\n")


if __name__ == "__main__":
    test_visual_emotion_loop()
