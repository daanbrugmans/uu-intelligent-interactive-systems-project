"""Emotion-aware cashier conversation system using LangGraph and Gemini.

Visual emotion recognition enables context-aware responses based on detected
valence/arousal. No text input from user - purely visual interaction loop.
Includes rate limiting and speech duration estimation for proper timing.
"""

import logging
import random
import time
from typing import Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from src.types import ConversationState, Emotion, EmotionType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MIN_API_DELAY_SECONDS = 0.5
FURHAT_WORDS_PER_SECOND = 2.5
POST_SPEECH_BUFFER_SECONDS = 0.5


class AgentState(TypedDict):
  """State container for visual emotion loop.
  
  Attributes:
    state: Conversation state with history and context.
    emotion: Current detected emotion from classifier.
    confidence: Model confidence in emotion prediction.
    continue_conversation: Whether to continue after this turn.
    speech_duration: Estimated speaking time for Furhat.
    customer_speech: Customer input from speech-to-text.
  """

  state: ConversationState
  emotion: Optional[Emotion]
  confidence: float
  continue_conversation: bool
  speech_duration: float
  customer_speech: str



class ConversationGraph:
  """Conversation system using LangGraph with emotion-based routing.
  
  Routes to positive/negative response paths based on emotion valence.
  Generates contextual responses via Gemini API with rate limiting.
  Provides timing information for loop synchronization.
  """

  EMOTION_PROMPTS = {
      EmotionType.POSITIVE: "warm, enthusiastic, and matching their positive energy",
      EmotionType.NEUTRAL: "calm, friendly, and professionally attentive",
      EmotionType.NEGATIVE: "gentle, understanding, and reassuring",
  }

  EMOTION_OBSERVATIONS = {
      EmotionType.POSITIVE: "smiling and looking engaged",
      EmotionType.NEUTRAL: "calm with a neutral expression",
      EmotionType.NEGATIVE: "looking uncertain or concerned",
  }

  CONVERSATION_TOPICS = [
      "weather",
      "season",
      "store",
      "checkout",
      "day",
      "products",
      "deals",
  ]

  STAGE_PROMPTS = {
      "introduction": """FIRST interaction - AI DISCLOSURE REQUIRED:
Greet warmly and state: (1) You are an AI, (2) You use emotion detection.
Example: Hi there! I'm an AI cashier - I use emotion detection to help serve you better. Welcome!""",
      "engagement": """Make natural small talk. Pick ONE topic (weather, season, day, shopping).
Keep it brief and natural - one or two sentences.""",
      "weather": """Continue small talk about weather or season.
One sentence, natural tone.""",
      "discounts": """Mention store promotions or special offers.
Keep it brief and offer-focused.""",
      "assistance": """Offer to help or check on their shopping.
Be helpful and friendly.""",
      "handling_negatives": """Customer didn't find something.
Look it up for them, be genuine and helpful. Sound natural and reassuring.""",
      "handling_negatives_result": """You looked it up. Choose ONE outcome:
OUTCOME A (Found): "Great news! I found it - that'll be {extra_cost} extra."
OUTCOME B (Out of stock): "Unfortunately it's out of stock, but we're getting more next week."
Pick whichever feels natural.""",
      "transition_to_payment": """Transition naturally to payment.
Example: "Ready to wrap up?" or "Let me ring this up for you.""",
      "payment": """Ask for payment method.
Say the total and ask cash or card.
Example: "Alright, your total is {total} SEK. Cash or card?""",
      "payment_processing": """Customer told payment method.
Confirm receipt or change handling. Brief response.""",
      "closing": """FINAL MESSAGE - Give receipt and end.
If CASH: "Here's your receipt and your change. Thank you, have a great day!"
If CARD: "Here's your receipt. Thank you, have a great day!""",
      "farewell": """Say goodbye briefly. Thank them. This is the end.""",
  }

  def __init__(self, api_key: str) -> None:
    """Initializes the conversation graph with Gemini LLM.
    
    Args:
      api_key: Google AI API key for Gemini access.
    
    Note:
      max_retries=0 disables automatic retries to prevent retry storms
      on rate limit (429) errors.
    """
    self._last_api_call_time = 0.0

    self.llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=api_key,
        temperature=0.8,
        max_output_tokens=80,
        max_retries=0,
    )

    self.graph = self._build_graph()
    logger.info("ConversationGraph initialized")

  def _build_graph(self) -> StateGraph:
    """Builds LangGraph workflow with emotion-based branching.
    
    Returns:
      Compiled StateGraph with nodes and edges configured.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("classify_emotion", self._classify_emotion)
    workflow.add_node("respond_positive", self._respond_positive)
    workflow.add_node("respond_negative", self._respond_negative)
    workflow.add_node("finalize", self._finalize)

    workflow.set_entry_point("classify_emotion")

    workflow.add_conditional_edges(
        "classify_emotion",
        self._route_by_emotion,
        {"positive": "respond_positive", "negative": "respond_negative"},
    )

    workflow.add_edge("respond_positive", "finalize")
    workflow.add_edge("respond_negative", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()

  def _route_by_emotion(
      self, state: AgentState
  ) -> Literal["positive", "negative"]:
    """Routes to response node based on emotion valence.
    
    Args:
      state: Current agent state.
    
    Returns:
      Node name: "positive" or "negative".
    """
    emotion = state.get("emotion")
    if emotion and emotion.emotion_type == EmotionType.POSITIVE:
      logger.debug(
          f"Routing to POSITIVE path (emotion: {emotion.emotion_type.name})"
      )
      return "positive"

    emotion_name = emotion.emotion_type.name if emotion else "NONE"
    logger.debug(f"Routing to NEGATIVE path (emotion: {emotion_name})")
    return "negative"

  def _classify_emotion(self, state: AgentState) -> AgentState:
    """Stores detected emotion in conversation history."""
    if state["emotion"]:
      state["state"].add_emotion(state["emotion"])
      logger.debug(
          f"Recorded emotion: {state['emotion'].emotion_type.name}"
      )
    return state

  def _respond_positive(self, state: AgentState) -> AgentState:
    """Generates response for positive emotions."""
    logger.info("Generating POSITIVE response")
    return self._generate_response(state, tone="warm and friendly")

  def _respond_negative(self, state: AgentState) -> AgentState:
    """Generates response for negative/neutral emotions."""
    logger.info("Generating EMPATHETIC response")
    return self._generate_response(state, tone="empathetic and helpful")
  def _generate_response(self, state: AgentState, tone: str) -> AgentState:
    """Generates response using Gemini with context from emotion history.
    
    Args:
      state: Current agent state with detected emotion.
      tone: Response tone (e.g., "warm and friendly").
    
    Returns:
      Updated state with agent_response and speech_duration.
    
    Note:
      Includes rate limiting and graceful fallback on API errors.
    """
    emotion = state["emotion"] or Emotion(EmotionType.NEUTRAL, 5.0, 5.0)
    confidence = state.get("confidence", 0.5)
    round_num = state["state"].conversation_round
    customer_speech = state.get("customer_speech", "")
    conv_state = state["state"]

    if conv_state.total_amount == 0:
      conv_state.total_amount = round(random.uniform(50, 599), 2)

    if customer_speech:
      speech_lower = customer_speech.lower()
      cash_keywords = [
          "cash",
          "cache",
          "cush",
          "catch",
          "kash",
          "cass",
          "cas",
      ]
      card_keywords = [
          "card",
          "credit",
          "debit",
          "visa",
          "tap",
          "cart",
          "cod",
      ]

      if any(kw in speech_lower for kw in cash_keywords) and not conv_state.payment_method:
        conv_state.payment_method = "cash"
        logger.info(
            f"Customer chose CASH payment (detected in: '{customer_speech}')"
        )
      elif (
          any(kw in speech_lower for kw in card_keywords)
          and not conv_state.payment_method
      ):
        conv_state.payment_method = "card"
        logger.info(
            f"Customer chose CARD payment (detected in: '{customer_speech}')"
        )
      conv_state.customer_speech = customer_speech

    stage_name = self._get_conversation_stage(round_num, conv_state)

    if stage_name == "payment":
      conv_state.asked_payment = True
      stage_prompt = self.STAGE_PROMPTS.get(stage_name, "").format(
          total=f"{conv_state.total_amount:.2f}"
      )
    elif stage_name == "closing":
      if conv_state.payment_method == "cash":
        stage_prompt = (
            "Give them receipt AND change. "
            "Say: Here's your receipt and your change. Thank them warmly!"
        )
      else:
        stage_prompt = (
            "Give them receipt. "
            "Say: Here's your receipt. Thank them warmly!"
        )
    else:
      stage_prompt = self.STAGE_PROMPTS.get(
          stage_name, self.STAGE_PROMPTS["engagement"]
      )

    emotion_style = self.EMOTION_PROMPTS.get(emotion.emotion_type, "friendly")
    emotion_observation = self.EMOTION_OBSERVATIONS.get(
        emotion.emotion_type, "present"
    )

    system_prompt = self._build_natural_prompt(
        tone,
        emotion_style,
        stage_name,
        stage_prompt,
        emotion_observation,
        state,
    )

    visual_observation = self._build_visual_observation(
        emotion, confidence, round_num
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=visual_observation),
    ]

    self._apply_rate_limit()

    try:
      logger.info(f"Calling Gemini API for round {round_num}...")
      response = self.llm.invoke(messages)
      state["state"].agent_response = response.content.strip()
      logger.info(f'Response: "{state["state"].agent_response}"')

    except Exception as e:
      error_msg = str(e)

      if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
        logger.warning("QUOTA EXHAUSTED - Using fallback (no retry)")
      elif "401" in error_msg or "403" in error_msg:
        logger.error("API KEY INVALID - Check your .env file")
      else:
        logger.error(f"LLM error: {e}")

      state["state"].agent_response = self._get_fallback_response(
          emotion.emotion_type, round_num, conv_state
      )
      logger.info(f'Fallback: "{state["state"].agent_response}"')

    state["speech_duration"] = self._estimate_speech_duration(
        state["state"].agent_response
    )

    return state
  def _build_visual_observation(
      self, emotion: Emotion, confidence: float, round_num: int
  ) -> str:
    """Builds the visual observation message describing what we see.
    
    Args:
      emotion: Detected emotion.
      confidence: Model confidence (0-1).
      round_num: Current conversation round.
    
    Returns:
      Observation string to guide the LLM response.
    """
    confidence_desc = "clearly" if confidence > 0.7 else "somewhat"
    emotion_name = emotion.emotion_type.name.lower()

    emotion_descriptions = {
        "positive": ["smiling", "looking cheerful", "in good spirits", "looking happy"],
        "neutral": ["calm", "relaxed", "at ease", "composed"],
        "negative": [
            "a bit tense",
            "looking thoughtful",
            "seeming preoccupied",
            "looking concerned",
        ],
    }

    desc_options = emotion_descriptions.get(emotion_name, ["present"])
    desc = random.choice(desc_options)

    if round_num == 0:
      return f"A customer just walked up. They look {desc}. Greet them with AI disclosure."
    elif round_num <= 2:
      return (
          f"The customer is {desc}. Make small talk - weather, the day, something natural."
      )
    elif round_num <= 4:
      return f"The customer looks {desc}. Continue chatting, maybe mention products or deals."
    elif round_num <= 6:
      return f"Conversation winding down. Customer is {desc}. Start wrapping up naturally."
    else:
      return f"Time to say goodbye. Customer looks {desc}. Be warm and brief."

  def _build_natural_prompt(
      self,
      tone: str,
      emotion_style: str,
      stage_name: str,
      stage_prompt: str,
      emotion_observation: str,
      state: AgentState,
  ) -> str:
    """Builds the system prompt with full conversation memory.
    
    Args:
      tone: Response tone.
      emotion_style: Emotion-specific style.
      stage_name: Current conversation stage.
      stage_prompt: Stage-specific instructions.
      emotion_observation: What we observe visually.
      state: Current agent state.
    
    Returns:
      Complete system prompt for the LLM.
    """
    num_turns = len(state["state"].conversation_history)

    prompt = (
        f"""You are a friendly store cashier having a NATURAL conversation.

YOU CAN SENSE EMOTIONS: You have emotion detection technology.
The customer is currently {emotion_observation}.
Your tone: {tone}
This is turn {num_turns + 1} of the conversation.

        === CURRENT TASK ({stage_name.upper()}) ===
        {stage_prompt}

        === CRITICAL RULES ===
        1. NEVER repeat something you already said - check history below
        2. Keep responses SHORT (1-2 sentences max)
        3. Be NATURAL - like a real person
        4. Do NOT introduce yourself again if you already did""

=== CRITICAL RULES ===
1. NEVER repeat something you already said - check history below
2. Keep responses SHORT (1-2 sentences max)
3. Be NATURAL - like a real person
4. Do NOT introduce yourself again if you already did
5. Be helpful and responsive
"""
    )

    if state["state"].conversation_history:
      prompt += "\n\n=== CONVERSATION SO FAR (DO NOT REPEAT ANY OF THIS) ==="
      for turn in state["state"].conversation_history:
        prompt += f'\nYou said: "{turn.agent_response}"'

      prompt += "\n\n^^^ DO NOT SAY ANY OF THE ABOVE AGAIN ^^^"

      all_responses = " ".join(
          [t.agent_response.lower() for t in state["state"].conversation_history]
      )
      covered_topics = []
      if "weather" in all_responses or "day out" in all_responses:
        covered_topics.append("weather")
      if "okay" in all_responses or "alright" in all_responses:
        covered_topics.append("asking if they're okay")
      if "find" in all_responses or "looking for" in all_responses:
        covered_topics.append("finding items")
      if "help" in all_responses:
        covered_topics.append("offering help")

      if covered_topics:
        prompt += (
            f"\n\nTOPICS ALREADY COVERED (pick something NEW): {', '.join(covered_topics)}"
        )

    if (
        state["state"].emotion_history
        and len(state["state"].emotion_history) > 1
    ):
      recent_emotions = state["state"].emotion_history[-3:]
      trend = [e.emotion_type.name for e in recent_emotions]
      prompt += f"\n\nEmotion journey so far: {' → '.join(trend)}"

      if trend[-1] != trend[0]:
        prompt += f" (their mood shifted from {trend[0]} to {trend[-1]})"

    customer_speech = state.get("customer_speech", "")
    if customer_speech:
      prompt += (
          f'\n\n=== CUSTOMER JUST SAID ===\n"{customer_speech}"\nRespond naturally!'
      )

    conv_state = state["state"]
    if stage_name in ("payment", "payment_processing", "closing"):
      if conv_state.total_amount > 0 and stage_name == "payment":
        prompt += f"\n\nTOTAL AMOUNT TO ANNOUNCE: {conv_state.total_amount:.2f} SEK"
      if conv_state.payment_method:
        prompt += f"\nCUSTOMER PAYING WITH: {conv_state.payment_method.upper()}"
        if conv_state.payment_method == "cash":
          prompt += " - Give receipt AND change!"
        else:
          prompt += " - Give receipt only!"
      if conv_state.asked_payment and not conv_state.payment_method:
        prompt += "\nWAITING FOR: Customer to say 'cash' or 'card'"

    return prompt

  def _estimate_speech_duration(self, text: str) -> float:
    """Estimates speech duration in seconds.
    
    Args:
      text: Text to be spoken.
    
    Returns:
      Estimated duration including buffer time.
    """
    word_count = len(text.split())
    duration = word_count / FURHAT_WORDS_PER_SECOND
    return duration + POST_SPEECH_BUFFER_SECONDS

  def _apply_rate_limit(self) -> None:
    """Ensures minimum delay between API calls."""
    current_time = time.time()
    time_since_last_call = current_time - self._last_api_call_time

    if time_since_last_call < MIN_API_DELAY_SECONDS:
      sleep_time = MIN_API_DELAY_SECONDS - time_since_last_call
      logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
      time.sleep(sleep_time)

    self._last_api_call_time = time.time()
  def _get_conversation_stage(
      self, round_num: int, state: ConversationState = None
  ) -> str:
    """Maps conversation round to stage name with strict progression.
    
    Args:
      round_num: Current conversation round number.
      state: Current conversation state.
    
    Returns:
      Stage name for current round.
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

    if state and state.payment_method:
      return "closing"

    if state and state.asked_payment and not state.payment_method:
      return "payment_processing"

    if state and state.looking_up_item:
      if state.item_found is None:
        return "handling_negatives_result"
      else:
        if state.item_found:
          state.total_amount += 15.0
        state.item_found = None
        state.looking_up_item = False

    if round_num < len(stages_sequence):
      stage = stages_sequence[round_num]

      if stage == "assistance" and state and state.customer_speech:
        speech_lower = state.customer_speech.lower()
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
          state.looking_up_item = True
          return "handling_negatives"

      return stage
    else:
      return "farewell"

  def _get_fallback_response(
      self,
      emotion_type: EmotionType,
      round_num: int,
      state: ConversationState = None,
  ) -> str:
    """Gets fallback response when LLM fails.
    
    Args:
      emotion_type: Detected emotion type.
      round_num: Current conversation round.
      state: Current conversation state.
    
    Returns:
      Fallback response string.
    """
    stage = self._get_conversation_stage(round_num, state)

    if stage == "introduction":
      responses = {
          EmotionType.POSITIVE: (
              "Hello and welcome! I'm an AI assistant using emotion "
              "detection to serve you better. I can see you're in great "
              "spirits today! How can I help you?"
          ),
          EmotionType.NEUTRAL: (
              "Hello! I'm an AI assistant that uses emotion detection "
              "technology to understand how you're feeling. What brings "
              "you in today?"
          ),
          EmotionType.NEGATIVE: (
              "Hi there, welcome. I'm an AI assistant that can sense "
              "emotions. Take your time, I'm here to help."
          ),
      }
    elif stage == "engagement":
      responses = {
          EmotionType.POSITIVE: "Nice weather we're having today, isn't it?",
          EmotionType.NEUTRAL: "Can you believe it's already January? Time flies!",
          EmotionType.NEGATIVE: "We just got some new items in if you want to take a look.",
      }
    elif stage == "weather":
      responses = {
          EmotionType.POSITIVE: "Beautiful day out there, I hope you're enjoying it!",
          EmotionType.NEUTRAL: "January weather keeps things interesting, doesn't it?",
          EmotionType.NEGATIVE: "Hope you found what you needed despite the weather!",
      }
    elif stage == "discounts":
      responses = {
          EmotionType.POSITIVE: (
              "By the way, we have some amazing deals running this week!"
          ),
          EmotionType.NEUTRAL: (
              "Just so you know, there are some great discounts on items today."
          ),
          EmotionType.NEGATIVE: (
              "Don't forget to check out our special offers - might find something great!"
          ),
      }
    elif stage == "assistance":
      responses = {
          EmotionType.POSITIVE: "Did you find everything you were looking for?",
          EmotionType.NEUTRAL: "Was there anything else you needed to find?",
          EmotionType.NEGATIVE: (
              "Did you locate everything alright? I'm here if you need anything."
          ),
      }
    elif stage == "handling_negatives":
      responses = {
          EmotionType.POSITIVE: (
              "No problem at all! Let me check the storage for you right now..."
          ),
          EmotionType.NEUTRAL: "Not a problem! Let me look that up for you.",
          EmotionType.NEGATIVE: (
              "I completely understand. Let me see what I can find for you."
          ),
      }
    elif stage == "handling_negatives_result":
      found = random.random() < 0.7
      if found:
        responses = {
            EmotionType.POSITIVE: (
                "Great news! I found it - that'll be 15 SEK extra, "
                "so your new total is {adjusted_total}."
            ),
            EmotionType.NEUTRAL: "Perfect! I found it. That'll be 15 SEK more for you.",
            EmotionType.NEGATIVE: (
                "Good news! I managed to find it for you. It's 15 SEK extra."
            ),
        }
        if state:
          state.total_amount += 15.0
          state.item_found = True
        response = responses.get(emotion_type, "Great! I found it!")
        return response.format(
            adjusted_total=f"{(state.total_amount if state else 315):.2f}"
        )
      else:
        reasons = [
            "Unfortunately it's out of stock, but we're getting more next week.",
            (
                "I'm sorry, we don't have that available right now, "
                "but we can order it for you."
            ),
            "That one's sold out at the moment, but it should be back soon.",
            "We don't currently have that in stock, but I can put you on the list.",
        ]
        if state:
          state.item_found = False
        return random.choice(reasons)
    elif stage == "transition_to_payment":
      responses = {
          EmotionType.POSITIVE: "Perfect! Let me ring this up for you.",
          EmotionType.NEUTRAL: "Alright, let me get that ready for you.",
          EmotionType.NEGATIVE: "Of course, let me help you check out.",
      }
    elif stage == "payment":
      total = state.total_amount if state else 299.00
      return f"Alright, your total comes to {total:.2f} SEK. Will that be cash or card?"
    elif stage == "payment_processing":
      return "Sorry, was that cash or card?"
    elif stage == "closing":
      if state and state.payment_method == "cash":
        return "Here's your receipt and your change. Thank you so much!"
      else:
        return "Here's your receipt. Thank you so much!"
    else:
      responses = {
          EmotionType.POSITIVE: "Thanks for stopping by! Have a wonderful day!",
          EmotionType.NEUTRAL: "Thanks for visiting! Take care!",
          EmotionType.NEGATIVE: "Thanks for coming in. Take care!",
      }

    return responses.get(emotion_type, "Thank you, have a great day!")

  def _finalize(self, state: AgentState) -> AgentState:
    """Records the conversation turn and checks for termination.
    
    Args:
      state: Current agent state.
    
    Returns:
      Updated agent state with termination flag.
    """
    state["state"].conversation_round += 1
    round_num = state["state"].conversation_round

    if state["emotion"]:
      visual_context = f"[Visual: {state['emotion'].emotion_type.name}]"
      state["state"].add_turn(
          user_input=visual_context,
          agent_response=state["state"].agent_response,
          emotion=state["emotion"],
      )

    should_end = state["state"].should_terminate(
        max_duration_seconds=120, max_rounds=10
    )
    state["continue_conversation"] = not should_end

    speech_dur = state.get("speech_duration", 0)
    logger.info(
        f"Round {round_num} complete | Speech: {speech_dur:.1f}s | "
        f"Continue: {not should_end}"
    )

    if should_end:
      logger.info("Conversation termination condition met")

    return state

  def invoke(self, state: AgentState) -> AgentState:
    """Executes one turn of the visual emotion loop.
    
    Args:
      state: AgentState with emotion from classifier.
    
    Returns:
      Updated AgentState with agent_response and speech_duration.
    
    Example:
      emotion, confidence = classifier.predict(frame)
      result = graph.invoke({
          "state": session,
          "emotion": emotion,
          "confidence": confidence,
          "continue_conversation": True,
          "speech_duration": 0.0
      })
      furhat.say(result["state"].agent_response)
      time.sleep(result["speech_duration"])
    """
    round_num = state["state"].conversation_round + 1
    emotion_name = (
        state["emotion"].emotion_type.name if state["emotion"] else "NONE"
    )
    confidence = state.get("confidence", 0.0)

    logger.info(
        f"Round {round_num} | Emotion: {emotion_name} ({confidence:.0%})"
    )
    return self.graph.invoke(state)
