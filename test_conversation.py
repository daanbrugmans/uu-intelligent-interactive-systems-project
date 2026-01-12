"""
Simple test for LangChain + LangGraph + Gemini conversation system
"""

import logging
from config.settings import Config
from src.conversation_graph import ConversationGraph, AgentState
from src.types import ConversationState, Emotion, EmotionType

logging.basicConfig(level=logging.INFO)


def test_conversation():
    """Test a simple conversation flow with different emotions"""
    
    print("=" * 60)
    print("LangChain + LangGraph + Gemini Test")
    print("=" * 60)
    
    # Validate config
    Config.validate()
    print(f"API Key: {Config.GEMINI_API_KEY[:20]}...")
    
    # Initialize graph
    graph = ConversationGraph(Config.GEMINI_API_KEY)
    print("Graph initialized successfully")
    print()
    
    # Test scenarios with 3 emotions: Happy, Neutral, Sad
    scenarios = [
        ("Hey there!", EmotionType.HAPPY, 8.0, 6.0),
        ("These items look nice", EmotionType.NEUTRAL, 7.0, 5.0),
        ("How much is this?", EmotionType.NEUTRAL, 7.0, 5.0),
        ("I'm not sure about this...", EmotionType.SAD, 3.0, 3.0),
        ("Thanks, goodbye!", EmotionType.HAPPY, 8.5, 5.5),
    ]
    
    session = ConversationState(session_id="test-001")
    
    for i, (user_input, emotion_type, valence, arousal) in enumerate(scenarios, 1):
        print(f"--- Turn {i} ---")
        print(f"Emotion: {emotion_type.name} (V:{valence}, A:{arousal})")
        print(f"User: {user_input}")
        
        session.user_input = user_input
        emotion = Emotion(emotion_type, valence, arousal)
        
        result = graph.invoke({
            "state": session,
            "emotion": emotion,
            "continue_conversation": True
        })
        
        session = result["state"]
        print(f"Bot: {session.agent_response}")
        print()
    
    print("=" * 60)
    print(f"Completed {len(scenarios)} turns successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_conversation()
