"""
Emotion-Aware Cashier Conversation System
==========================================

Visual Emotion Loop:
    Webcam → EmotionClassifier → ConversationGraph → Furhat Speech

Components:
    - EmotionClassifier: ConvNeXt-Tiny model for valence/arousal prediction
    - ConversationGraph: LangGraph workflow with Gemini LLM
    - Types: Emotion, EmotionType, ConversationState
"""

from src.types import ConversationState, Emotion, EmotionType, ConversationTurn
from src.conversation_graph import ConversationGraph, AgentState
from src.emotion_classifier import EmotionClassifier, ClassifierConfig

__all__ = [
    # Emotion Classification
    "EmotionClassifier",
    "ClassifierConfig",
    
    # Conversation Graph
    "ConversationGraph",
    "AgentState",
    
    # Data Types
    "ConversationState",
    "Emotion",
    "EmotionType",
    "ConversationTurn",
]
