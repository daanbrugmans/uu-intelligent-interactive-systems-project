from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class EmotionType(Enum):
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"


@dataclass
class Emotion:
    emotion_type: EmotionType
    valence: float
    arousal: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConversationTurn:
    """Single conversation turn with emotion context"""

    user_input: str
    agent_response: str
    emotion: Emotion
    round_number: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConversationState:
    session_id: str
    current_emotion: Optional[Emotion] = None
    emotion_history: list = field(default_factory=list)
    conversation_history: list = field(default_factory=list)  # List of ConversationTurn
    conversation_round: int = 0
    user_input: str = ""
    agent_response: str = ""
    is_complete: bool = False
    started_at: datetime = field(default_factory=datetime.now)

    def add_emotion(self, emotion: Emotion):
        self.emotion_history.append(emotion)
        self.current_emotion = emotion

    def add_turn(self, user_input: str, agent_response: str, emotion: Emotion):
        """Record a complete conversation turn with emotion context"""
        turn = ConversationTurn(
            user_input=user_input,
            agent_response=agent_response,
            emotion=emotion,
            round_number=self.conversation_round,
        )
        self.conversation_history.append(turn)
        self.user_input = user_input
        self.agent_response = agent_response
        self.current_emotion = emotion

    def get_last_turn(self) -> Optional[ConversationTurn]:
        """Get the most recent conversation turn"""
        return self.conversation_history[-1] if self.conversation_history else None

    def elapsed_seconds(self) -> float:
        return (datetime.now() - self.started_at).total_seconds()

    def should_terminate(self, max_time: int = 120, max_rounds: int = 10) -> bool:
        if self.elapsed_seconds() > max_time:
            return True
        if self.conversation_round >= max_rounds:
            return True
        return False
