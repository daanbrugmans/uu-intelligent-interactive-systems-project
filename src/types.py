"""Type definitions for emotion-aware conversation system.

Core data structures for webcam-based emotion detection and visual-only
conversation loop with Gemini LLM integration.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


class EmotionType(Enum):
  """Supported emotion types from ConvNeXt classifier."""

  POSITIVE = "Positive"
  NEUTRAL = "Neutral"
  NEGATIVE = "Negative"


@dataclass
class Emotion:
  """Single emotion detection from webcam frame.
  
  Attributes:
    emotion_type: Classified emotion category.
    valence: Positive/negative dimension (1-10 scale).
    arousal: Energy/activation level (1-10 scale).
    timestamp: When emotion was detected.
  """

  emotion_type: EmotionType
  valence: float
  arousal: float
  timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConversationTurn:
  """Single round of the conversation loop.
  
  Attributes:
    user_input: Visual context description (e.g., "[Visual: HAPPY]").
    agent_response: Gemini-generated text for Furhat.
    emotion: Detected emotion for this turn.
    round_number: Conversation round index.
    timestamp: When this turn occurred.
  """

  user_input: str
  agent_response: str
  emotion: Emotion
  round_number: int
  timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConversationState:
  """Full state of a conversation session.
  
  Tracks customer interaction from arrival until departure or timeout.
  Supports payment processing and item lookup flows.
  
  Attributes:
    session_id: Unique session identifier.
    current_emotion: Most recently detected emotion.
    emotion_history: All emotions detected during session.
    conversation_history: All turns with agent responses.
    conversation_round: Current round number (0-indexed).
    user_input: Visual context from current turn.
    agent_response: Most recent agent response.
    is_complete: Whether session has ended.
    started_at: Session start timestamp.
    payment_method: Payment type ("cash" or "card").
    total_amount: Total in SEK.
    customer_speech: Customer input from speech-to-text.
    asked_payment: Whether payment method was requested.
    looking_up_item: Whether looking up missing item.
    item_found: Result of item lookup (True/False/None).
  
  Termination:
    - Max duration: 120 seconds
    - Max rounds: 10 turns
  """

  session_id: str = ""
  current_emotion: Optional[Emotion] = None
  emotion_history: List[Emotion] = field(default_factory=list)
  conversation_history: List[ConversationTurn] = field(
      default_factory=list
  )
  conversation_round: int = 0
  user_input: str = ""
  agent_response: str = ""
  is_complete: bool = False
  started_at: datetime = field(default_factory=datetime.now)

  payment_method: Optional[str] = None
  total_amount: float = 0.0
  customer_speech: str = ""
  asked_payment: bool = False

  looking_up_item: bool = False
  item_found: Optional[bool] = None

  def add_emotion(self, emotion: Emotion) -> None:
    """Records a detected emotion in history.
    
    Args:
      emotion: Emotion object to record.
    """
    self.emotion_history.append(emotion)
    self.current_emotion = emotion

  def add_turn(
      self, user_input: str, agent_response: str, emotion: Emotion
  ) -> None:
    """Records a complete conversation turn.
    
    Args:
      user_input: Visual context description.
      agent_response: Gemini-generated response.
      emotion: Detected emotion for this turn.
    """
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
    """Returns the most recent conversation turn.
    
    Returns:
      Last ConversationTurn or None if no history.
    """
    return (
        self.conversation_history[-1]
        if self.conversation_history
        else None
    )

  def elapsed_seconds(self) -> float:
    """Returns seconds elapsed since session start.
    
    Returns:
      Elapsed time in seconds.
    """
    return (datetime.now() - self.started_at).total_seconds()

  def should_terminate(
      self,
      max_duration_seconds: int = 120,
      max_rounds: int = 10,
  ) -> bool:
    """Checks if session should end.
    
    Args:
      max_duration_seconds: Maximum session duration in seconds.
      max_rounds: Maximum conversation rounds.
    
    Returns:
      True if termination condition met.
    """
    if self.elapsed_seconds() > max_duration_seconds:
      return True
    if self.conversation_round >= max_rounds:
      return True
    return False

  def get_emotion_trend(self, last_n: int = 3) -> List[EmotionType]:
    """Gets recent emotion trend.
    
    Args:
      last_n: Number of recent emotions to include.
    
    Returns:
      List of recent emotion types.
    """
    recent = self.emotion_history[-last_n:] if self.emotion_history else []
    return [e.emotion_type for e in recent]
