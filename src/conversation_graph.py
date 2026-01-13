"""
Emotion-Aware Cashier Conversation System
==========================================


Key Design Decisions:
    - NO text input from user - purely visual emotion recognition
    - Conversation context inferred from emotion history + round number
    - Rate limiting prevents API quota exhaustion
    - Speech duration estimation for proper timing
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, Literal
from src.types import ConversationState, EmotionType, Emotion
import logging
import time

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# TIMING CONSTANTS
# ============================================================================

# Minimum delay between Gemini API calls (prevents quota exhaustion)
MIN_API_DELAY_SECONDS = 0.5

# Estimated speech rate for Furhat (words per second)
FURHAT_WORDS_PER_SECOND = 2.5

# Buffer time after speech before next frame capture
POST_SPEECH_BUFFER_SECONDS = 0.5


# ============================================================================
# STATE DEFINITIONS
# ============================================================================

class AgentState(TypedDict):
    """
    LangGraph state container for visual emotion loop.
    
    Attributes:
        state: The conversation state with history and context
        emotion: Current detected emotion from webcam/ConvNeXt
        confidence: Model confidence in the emotion prediction
        continue_conversation: Whether to continue after this turn
        speech_duration: Estimated time for Furhat to speak response
        customer_speech: What the customer said (from speech-to-text)
    """
    state: ConversationState
    emotion: Optional[Emotion]
    confidence: float
    continue_conversation: bool
    speech_duration: float
    customer_speech: str  # Speech-to-text result


# ============================================================================
# CONVERSATION GRAPH
# ============================================================================

class ConversationGraph:
    """
    LangGraph-based conversation system for visual emotion loop.
    
    This system:
        1. Receives emotion from webcam + ConvNeXt classifier
        2. Routes to positive/negative response path based on emotion
        3. Generates contextual response via Gemini (based on round + emotion history)
        4. Returns text for Furhat speech synthesis
        5. Provides timing information for loop synchronization
    
    Conversation Stages:
        Round 0: INTRODUCTION - Explain the system, greet customer
        Round 1-2: ENGAGEMENT - Build rapport, react to emotions
        Round 3-4: ASSISTANCE - Offer help based on emotional state
        Round 5-6: CLOSING - Wrap up naturally
        Round 7+: FAREWELL - Say goodbye
    
    Usage:
        graph = ConversationGraph(api_key="your-gemini-key")
        
        # Each round of the visual loop:
        result = graph.invoke({
            "state": conversation_state,
            "emotion": Emotion(EmotionType.POSITIVE, 8.0, 6.0),
            "confidence": 0.87,
            "continue_conversation": True,
            "speech_duration": 0.0
        })
        
        # Get response and timing
        text = result["state"].agent_response
        wait_time = result["speech_duration"]
    """
    
    # System prompt styles for each emotion type
    EMOTION_PROMPTS = {
        EmotionType.POSITIVE: "warm, enthusiastic, and matching their positive energy",
        EmotionType.NEUTRAL: "calm, friendly, and professionally attentive",
        EmotionType.NEGATIVE: "gentle, understanding, and reassuring"
    }
    
    # What we observe visually for each emotion
    EMOTION_OBSERVATIONS = {
        EmotionType.POSITIVE: "smiling and looking engaged",
        EmotionType.NEUTRAL: "calm with a neutral expression",
        EmotionType.NEGATIVE: "looking uncertain or concerned"
    }
    
    # Natural conversation topics to rotate through (prevents repetition)
    CONVERSATION_TOPICS = [
        "weather",      # "Nice weather we're having!"
        "season",       # "Can you believe it's already January?"
        "store",        # "We just got some new items in..."
        "checkout",     # "Ready to check out when you are"
        "day",          # "How's your day going?"
        "products",     # "Find everything you were looking for?"
        "deals",        # "We have some great deals today"
    ]
    
    # Stage-specific prompts for natural conversation flow
    STAGE_PROMPTS = {
        "introduction": """FIRST interaction - AI DISCLOSURE REQUIRED:
Greet them warmly and clearly state:
1. You are an AI assistant
2. You use emotion detection technology to understand their mood
3. This helps you serve them better

Example: Hi there! I'm an AI cashier - I use emotion detection to help serve you better. Welcome!""",
        
        "engagement": """Make natural small talk. Pick ONE topic:
- Comment on the weather or season
- Ask how their day is going
- Mention something about shopping

Keep it brief and natural - one or two sentences.""",
        
        "weather": """Continue small talk about weather or season:
- Comment on current weather
- Or ask about their day
- Keep it friendly and conversational

One sentence, natural tone.""",
        
        "discounts": """Mention store promotions or special offers:
- We have great deals today
- Special items on sale
- Check out the discounts

Keep it brief and offer-focused.""",
        
        "assistance": """Offer to help or check on their shopping:
- Did you find everything you were looking for?
- Need help finding anything?
- Everything okay with your items?

Be helpful and friendly.""",
        
        "handling_negatives": """Customer said they didn't find or get something.
YOU: Look it up for them, be genuine and helpful:
- "I'll look that up for you right now..."
- "Let me check if we have that in the back..."
- "One moment, let me see what I can find..."

Sound natural and reassuring, like you really care about helping them.""",
        
        "handling_negatives_result": """You looked it up. Choose ONE outcome:
OUTCOME A (Item found): "Great news! I found it - that'll be {extra_cost} extra."
OUTCOME B (Out of stock): "Unfortunately it's out of stock, but we're getting more next week."

Pick whichever feels more natural. If found, mention extra cost. If not, reassure them.
Be brief and conversational, not robotic.""",
        
        "transition_to_payment": """Transition naturally to payment:
- Ready to wrap up?
- Let me ring this up for you
- That will be all? Ready to pay?

Smooth, natural transition.""",
        
        "payment": """TIME TO ASK FOR PAYMENT:
Say the total and ask for payment method.
Example: "Alright, your total is {total} SEK. Cash or card?"
MUST include the total and ask cash or card.""",
        
        "payment_processing": """They told you their payment method.
If CASH: "Great, one moment..." (you will give change in next step)
If CARD: "Perfect, payment received!" (card processed instantly)
If unclear: Ask again "Sorry, cash or card?"
Brief response - do NOT wait for more input.""",
        
        "closing": """FINAL MESSAGE - Give receipt and end:
If they paid CASH: "Here's your receipt and your change. Thank you, have a great day!"
If they paid CARD: "Here's your receipt. Thank you, have a great day!"

THIS IS THE LAST MESSAGE. Include goodbye and thanks. Do not ask questions.""",
        
        "farewell": """Say goodbye briefly. Thank them. This is the end."""
    }
    
    def __init__(self, api_key: str):
        """
        Initialize the conversation graph with Gemini LLM.
        
        Args:
            api_key: Google AI API key for Gemini access
        
        Important:
            - max_retries=0 disables automatic retries to prevent retry storms
            - If quota exhausted, fails fast with fallback instead of 6+ retries
        """
        self._last_api_call_time = 0.0
        
        # CRITICAL: Disable retries to prevent retry storm on 429 errors
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=api_key,
            temperature=0.8,  # Slightly higher for more natural responses
            max_output_tokens=80,  # Allow slightly longer for natural speech
            max_retries=0,  # NO automatic retries - prevents quota exhaustion
        )
        
        self.graph = self._build_graph()
        logger.info("✅ ConversationGraph initialized (natural conversation mode)")
    
    # ========================================================================
    # GRAPH BUILDING
    # ========================================================================
    
    def _build_graph(self) -> StateGraph:
        """
        Build LangGraph workflow with emotion-based branching.
        
        Graph Structure:
            classify_emotion → (route) → respond_positive OR respond_negative
                                              ↓
                                          finalize → END
        """
        workflow = StateGraph(AgentState)
        
        # Register nodes
        workflow.add_node("classify_emotion", self._classify_emotion)
        workflow.add_node("respond_positive", self._respond_positive)
        workflow.add_node("respond_negative", self._respond_negative)
        workflow.add_node("finalize", self._finalize)
        
        # Set entry point
        workflow.set_entry_point("classify_emotion")
        
        # Conditional branching: route based on emotion valence
        workflow.add_conditional_edges(
            "classify_emotion",
            self._route_by_emotion,
            {
                "positive": "respond_positive",
                "negative": "respond_negative"
            }
        )
        
        # Both paths converge at finalize
        workflow.add_edge("respond_positive", "finalize")
        workflow.add_edge("respond_negative", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    # ========================================================================
    # ROUTING LOGIC
    # ========================================================================
    
    def _route_by_emotion(self, state: AgentState) -> Literal["positive", "negative"]:
        """
        Route to response node based on emotion valence.
        
        Routing Rules:
            - HAPPY → positive path (energetic response)
            - SAD, NEUTRAL → negative path (empathetic/calm response)
        """
        emotion = state.get("emotion")
        if emotion and emotion.emotion_type == EmotionType.POSITIVE:
            logger.debug(f"🟢 Routing to POSITIVE path (emotion: {emotion.emotion_type.name})")
            return "positive"
        
        emotion_name = emotion.emotion_type.name if emotion else "NONE"
        logger.debug(f"🔵 Routing to NEGATIVE path (emotion: {emotion_name})")
        return "negative"
    
    # ========================================================================
    # NODE IMPLEMENTATIONS
    # ========================================================================
    
    def _classify_emotion(self, state: AgentState) -> AgentState:
        """Store detected emotion in conversation history."""
        if state["emotion"]:
            state["state"].add_emotion(state["emotion"])
            logger.debug(f"📊 Recorded emotion: {state['emotion'].emotion_type.name}")
        return state
    
    def _respond_positive(self, state: AgentState) -> AgentState:
        """Generate response for positive emotions (POSITIVE)."""
        logger.info("💚 Generating POSITIVE response")
        return self._generate_response(state, tone="warm and friendly")
    
    def _respond_negative(self, state: AgentState) -> AgentState:
        """Generate response for negative/neutral emotions (NEGATIVE, NEUTRAL)."""
        logger.info("💙 Generating EMPATHETIC response")
        return self._generate_response(state, tone="empathetic and helpful")
    
    # ========================================================================
    # RESPONSE GENERATION (with rate limiting)
    # ========================================================================
    
    def _generate_response(self, state: AgentState, tone: str) -> AgentState:
        """
        Generate LLM response using LangChain + Gemini.
        
        This is a VISUAL-ONLY system - no text input from user.
        Context is derived from:
            - Current detected emotion
            - Emotion history (changes over time)
            - Conversation round number (stage)
            - Model confidence level
        
        Features:
            - Rate limiting (minimum delay between calls)
            - No automatic retries (fail fast on 429)
            - Graceful fallback on any error
            - Speech duration estimation for timing
        
        Args:
            state: Current agent state with emotion
            tone: Response tone (e.g., "warm and friendly")
        
        Returns:
            Updated state with agent_response and speech_duration
        """
        # Extract context
        emotion = state["emotion"] or Emotion(EmotionType.NEUTRAL, 5.0, 5.0)
        confidence = state.get("confidence", 0.5)
        round_num = state["state"].conversation_round
        customer_speech = state.get("customer_speech", "")
        conv_state = state["state"]
        
        # Generate random total if not set (between 50-599 SEK)
        if conv_state.total_amount == 0:
            import random
            conv_state.total_amount = round(random.uniform(50, 599), 2)
        
        # Check if customer said "cash" or "card" in their speech
        # Include common misheard variations for better recognition
        if customer_speech:
            speech_lower = customer_speech.lower()
            # Cash variations: "cash", "cache", "cush", "catch", "kash"
            cash_keywords = ["cash", "cache", "cush", "catch", "kash", "cass", "cas"]
            # Card variations: "card", "credit", "debit", "visa", "tap"
            card_keywords = ["card", "credit", "debit", "visa", "tap", "cart", "cod"]
            
            if any(kw in speech_lower for kw in cash_keywords) and not conv_state.payment_method:
                conv_state.payment_method = "cash"
                logger.info(f"💵 Customer chose CASH payment (detected in: '{customer_speech}')")
            elif any(kw in speech_lower for kw in card_keywords) and not conv_state.payment_method:
                conv_state.payment_method = "card"
                logger.info(f"💳 Customer chose CARD payment (detected in: '{customer_speech}')")
            # Store what customer said
            conv_state.customer_speech = customer_speech
        
        # Determine conversation stage (payment-aware)
        stage_name = self._get_conversation_stage(round_num, conv_state)
        
        # Handle payment stage - inject total amount
        if stage_name == "payment":
            conv_state.asked_payment = True
            stage_prompt = self.STAGE_PROMPTS.get(stage_name, "").format(
                total=f"{conv_state.total_amount:.2f}"
            )
        elif stage_name == "closing":
            # Customize closing based on payment method
            if conv_state.payment_method == "cash":
                stage_prompt = "Give them their receipt AND change. Say: Here's your receipt and your change. Thank them warmly!"
            else:
                stage_prompt = "Give them their receipt. Say: Here's your receipt. Thank them warmly!"
        else:
            stage_prompt = self.STAGE_PROMPTS.get(stage_name, self.STAGE_PROMPTS["engagement"])
        
        # Get emotion-specific style
        emotion_style = self.EMOTION_PROMPTS.get(emotion.emotion_type, "friendly")
        emotion_observation = self.EMOTION_OBSERVATIONS.get(
            emotion.emotion_type, 
            "present"
        )
        
        # Build the natural conversation prompt
        system_prompt = self._build_natural_prompt(
            tone, emotion_style, stage_name, stage_prompt, emotion_observation, state
        )
        
        # The "user message" describes what we see
        visual_observation = self._build_visual_observation(emotion, confidence, round_num)
        
        # Prepare messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=visual_observation)
        ]
        
        # Apply rate limiting
        self._apply_rate_limit()
        
        # Call Gemini (with no retries - fail fast)
        try:
            logger.info(f" Calling Gemini API for round {round_num}...")
            response = self.llm.invoke(messages)
            state["state"].agent_response = response.content.strip()
            logger.info(f" Response: \"{state['state'].agent_response}\"")
            
        except Exception as e:
            error_msg = str(e)
            
            # Identify error type for clear logging
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                logger.warning(" QUOTA EXHAUSTED - Using fallback (no retry)")
            elif "401" in error_msg or "403" in error_msg:
                logger.error("API KEY INVALID - Check your .env file")
            else:
                logger.error(f"❌ LLM error: {e}")
            
            # Use fallback instead of retrying
            state["state"].agent_response = self._get_fallback_response(
                emotion.emotion_type, round_num, conv_state
            )
            logger.info(f"🔄 Fallback: \"{state['state'].agent_response}\"")
        
        # Calculate speech duration for Furhat timing
        state["speech_duration"] = self._estimate_speech_duration(
            state["state"].agent_response
        )
        
        return state
    
    def _build_visual_observation(
        self, 
        emotion: Emotion, 
        confidence: float,
        round_num: int
    ) -> str:
        """
        Build the visual observation message.
        
        Describes what we see visually to guide the LLM response.
        """
        confidence_desc = "clearly" if confidence > 0.7 else "somewhat"
        emotion_name = emotion.emotion_type.name.lower()
        
        # More varied observations based on emotion
        emotion_descriptions = {
            "positive": ["smiling", "looking cheerful", "in good spirits", "looking happy"],
            "neutral": ["calm", "relaxed", "at ease", "composed"],
            "negative": ["a bit tense", "looking thoughtful", "seeming preoccupied", "looking concerned"]
        }
        
        import random
        desc_options = emotion_descriptions.get(emotion_name, ["present"])
        desc = random.choice(desc_options)
        
        if round_num == 0:
            return f"A customer just walked up. They look {desc}. Greet them with AI disclosure."
        elif round_num <= 2:
            return f"The customer is {desc}. Make small talk - weather, the day, something natural."
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
        state: AgentState
    ) -> str:
        """Build a natural conversation prompt with full memory."""
        num_turns = len(state["state"].conversation_history)
        
        prompt = f"""You are a friendly store cashier having a NATURAL conversation.

YOU CAN SENSE EMOTIONS: You have emotion detection technology - you can see how the customer feels.
The customer is currently {emotion_observation}.
Your tone: {tone}
This is turn {num_turns + 1} of the conversation.

=== CURRENT TASK ({stage_name.upper()}) ===
{stage_prompt}

=== CRITICAL RULES ===
1. NEVER repeat something you already said - check history below
2. Keep responses SHORT (1-2 sentences max)
3. Be NATURAL - like a real person
4. Do NOT introduce yourself again if you already did
5. if you are assisting the user bya sking a question di you find everything you were looking for , and they ask  you a question"""
        
        # Add FULL conversation history to prevent repetition
        if state["state"].conversation_history:
            prompt += "\n\n=== CONVERSATION SO FAR (DO NOT REPEAT ANY OF THIS) ==="
            for i, turn in enumerate(state["state"].conversation_history):
                emotion_name = turn.emotion.emotion_type.name if turn.emotion else "UNKNOWN"
                prompt += f"\nYou said: \"{turn.agent_response}\""
            
            prompt += "\n\n^^^ DO NOT SAY ANY OF THE ABOVE AGAIN ^^^"
            
            # Extract topics already covered
            all_responses = " ".join([t.agent_response.lower() for t in state["state"].conversation_history])
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
                prompt += f"\n\nTOPICS ALREADY COVERED (pick something NEW): {', '.join(covered_topics)}"
        
        # Add emotion trend
        if state["state"].emotion_history and len(state["state"].emotion_history) > 1:
            recent_emotions = state["state"].emotion_history[-3:]
            trend = [e.emotion_type.name for e in recent_emotions]
            prompt += f"\n\nEmotion journey so far: {' → '.join(trend)}"
            
            # Note if emotion changed
            if trend[-1] != trend[0]:
                prompt += f" (their mood shifted from {trend[0]} to {trend[-1]})"
        
        # Add customer speech if available
        customer_speech = state.get("customer_speech", "")
        if customer_speech:
            prompt += f"\n\n=== CUSTOMER JUST SAID ===\n\"{customer_speech}\"\nRespond naturally to what they said!"
        
        # Add payment context ONLY in payment/closing stages (not before)
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
        """
        Estimate how long Furhat will take to speak the response.
        
        Returns:
            Estimated duration in seconds
        """
        word_count = len(text.split())
        duration = word_count / FURHAT_WORDS_PER_SECOND
        return duration + POST_SPEECH_BUFFER_SECONDS
    
    def _apply_rate_limit(self) -> None:
        """
        Ensure minimum delay between API calls.
        
        This prevents accidental burst requests that could hit rate limits.
        """
        current_time = time.time()
        time_since_last_call = current_time - self._last_api_call_time
        
        if time_since_last_call < MIN_API_DELAY_SECONDS:
            sleep_time = MIN_API_DELAY_SECONDS - time_since_last_call
            logger.debug(f"⏱️ Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self._last_api_call_time = time.time()
    
    def _get_conversation_stage(self, round_num: int, state: ConversationState = None) -> str:
        """
        Map conversation round to stage name - STRICT, FOOLPROOF progression.
        
        This enforces a rigid stage sequence that CANNOT be interrupted or skipped,
        ensuring consistent conversation flow regardless of customer speech or emotion.
        """
        # STRICT SEQUENCE - follows this ALWAYS unless payment method is confirmed
        stages_sequence = [
            "introduction",          # Round 0: Greet + AI disclosure
            "engagement",            # Round 1: Small talk (day/weather)
            "weather",               # Round 2: Weather or season comment
            "discounts",             # Round 3: Mention special offers
            "assistance",            # Round 4: Ask if found everything
            "transition_to_payment", # Round 5: Smooth transition to payment
            "payment",               # Round 6: Ask cash or card + total
            "payment_processing",    # Round 7: Confirm payment method
            "closing",               # Round 8: Receipt + goodbye
            "farewell"               # Round 9+: Final goodbye
        ]
        
        # Handle payment method confirmation (only override after asking for payment)
        if state and state.payment_method:
            # Customer already chose cash/card, skip to closing
            return "closing"
        
        # If waiting for payment method (asked but not received), stay in payment_processing
        if state and state.asked_payment and not state.payment_method:
            return "payment_processing"
        
        # Handle item lookup flow (only occurs during assistance stage)
        if state and state.looking_up_item:
            if state.item_found is None:
                # Still looking up
                return "handling_negatives_result"
            else:
                # Lookup complete - add cost if found
                if state.item_found:
                    state.total_amount += 15.0
                # Reset lookup flags
                state.item_found = None
                state.looking_up_item = False
                # Continue with normal sequence (don't skip rounds)
        
        # STRICT: Use only round_num to determine stage
        # This ensures conversation always follows the sequence
        if round_num < len(stages_sequence):
            stage = stages_sequence[round_num]
            
            # Special: Check for "no" only at ASSISTANCE stage (round 4)
            # This triggers item lookup flow WITHIN the same round
            if stage == "assistance" and state and state.customer_speech:
                speech_lower = state.customer_speech.lower()
                no_keywords = ["no", "didn't", "don't", "nope", "nah", "didn't find", "don't have", "can't find", "couldn't find", "don't see"]
                if any(kw in speech_lower for kw in no_keywords):
                    state.looking_up_item = True
                    return "handling_negatives"  # Offer to look it up
            
            return stage
        else:
            # Beyond sequence length, return farewell
            return "farewell"
    
    def _get_fallback_response(self, emotion_type: EmotionType, round_num: int, state: ConversationState = None) -> str:
        """
        Get fallback response when LLM fails.
        
        Stage-appropriate responses that feel natural.
        Includes AI disclosure in introduction for regulatory compliance.
        """
        stage = self._get_conversation_stage(round_num, state)
        
        if stage == "introduction":
            # IMPORTANT: AI disclosure for regulatory compliance
            responses = {
                EmotionType.POSITIVE: "Hello and welcome! I'm an AI assistant using emotion detection to serve you better. I can see you're in great spirits today! How can I help you?",
                EmotionType.NEUTRAL: "Hello! I'm an AI assistant that uses emotion detection technology to understand how you're feeling. What brings you in today?",
                EmotionType.NEGATIVE: "Hi there, welcome. I'm an AI assistant that can sense emotions. Take your time, I'm here to help."
            }
        elif stage == "engagement":
            responses = {
                EmotionType.POSITIVE: "Nice weather we're having today, isn't it?",
                EmotionType.NEUTRAL: "Can you believe it's already January? Time flies!",
                EmotionType.NEGATIVE: "We just got some new items in if you want to take a look."
            }
        elif stage == "weather":
            responses = {
                EmotionType.POSITIVE: "Beautiful day out there, I hope you're enjoying it!",
                EmotionType.NEUTRAL: "January weather keeps things interesting, doesn't it?",
                EmotionType.NEGATIVE: "Hope you found what you needed despite the weather!"
            }
        elif stage == "discounts":
            responses = {
                EmotionType.POSITIVE: "By the way, we have some amazing deals running this week!",
                EmotionType.NEUTRAL: "Just so you know, there are some great discounts on items today.",
                EmotionType.NEGATIVE: "Don't forget to check out our special offers - might find something great!"
            }
        elif stage == "assistance":
            responses = {
                EmotionType.POSITIVE: "Did you find everything you were looking for?",
                EmotionType.NEUTRAL: "Was there anything else you needed to find?",
                EmotionType.NEGATIVE: "Did you locate everything alright? I'm here if you need anything."
            }
        elif stage == "handling_negatives":
            responses = {
                EmotionType.POSITIVE: "No problem at all! Let me check the storage for you right now...",
                EmotionType.NEUTRAL: "Not a problem! Let me look that up for you.",
                EmotionType.NEGATIVE: "I completely understand. Let me see what I can find for you."
            }
        elif stage == "handling_negatives_result":
            # Randomly decide if item was found (70% found, 30% not found)
            import random
            found = random.random() < 0.7
            if found:
                responses = {
                    EmotionType.POSITIVE: "Great news! I found it - that'll be 15 SEK extra, so your new total is {adjusted_total}.",
                    EmotionType.NEUTRAL: "Perfect! I found it. That'll be 15 SEK more for you.",
                    EmotionType.NEGATIVE: "Good news! I managed to find it for you. It's 15 SEK extra."
                }
                # Format with adjusted total
                if state:
                    state.total_amount += 15.0
                    state.item_found = True
                response = responses.get(emotion_type, "Great! I found it!")
                return response.format(adjusted_total=f"{(state.total_amount if state else 315):.2f}")
            else:
                # Item not found - give reasons
                reasons = [
                    "Unfortunately it's out of stock, but we're getting more next week.",
                    "I'm sorry, we don't have that available right now, but we can order it for you.",
                    "That one's sold out at the moment, but it should be back soon.",
                    "We don't currently have that in stock, but I can put you on the list."
                ]
                responses = {
                    EmotionType.POSITIVE: random.choice(reasons),
                    EmotionType.NEUTRAL: random.choice(reasons),
                    EmotionType.NEGATIVE: random.choice(reasons)
                }
                if state:
                    state.item_found = False
                return responses.get(emotion_type, random.choice(reasons))
        elif stage == "transition_to_payment":
            responses = {
                EmotionType.POSITIVE: "Perfect! Let me ring this up for you.",
                EmotionType.NEUTRAL: "Alright, let me get that ready for you.",
                EmotionType.NEGATIVE: "Of course, let me help you check out."
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
        else:  # farewell
            responses = {
                EmotionType.POSITIVE: "Thanks for stopping by! Have a wonderful day!",
                EmotionType.NEUTRAL: "Thanks for visiting! Take care!",
                EmotionType.NEGATIVE: "Thanks for coming in. Take care!"
            }
        
        return responses.get(emotion_type, "Thank you, have a great day!")
    
    # ========================================================================
    # FINALIZATION
    # ========================================================================
    
    def _finalize(self, state: AgentState) -> AgentState:
        """
        Record the conversation turn and check for termination.
        
        Also logs timing information for synchronization.
        """
        state["state"].conversation_round += 1
        round_num = state["state"].conversation_round
        
        # Record turn (with visual context instead of text input)
        if state["emotion"]:
            visual_context = f"[Visual: {state['emotion'].emotion_type.name}]"
            state["state"].add_turn(
                user_input=visual_context,
                agent_response=state["state"].agent_response,
                emotion=state["emotion"]
            )
        
        # Check termination conditions
        should_end = state["state"].should_terminate(
            max_duration_seconds=120,
            max_rounds=10
        )
        state["continue_conversation"] = not should_end
        
        # Log timing info
        speech_dur = state.get("speech_duration", 0)
        logger.info(f"📍 Round {round_num} complete | Speech: {speech_dur:.1f}s | Continue: {not should_end}")
        
        if should_end:
            logger.info("🏁 Conversation termination condition met")
        
        return state
    
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    def invoke(self, state: AgentState) -> AgentState:
        """
        Execute one turn of the visual emotion loop.
        
        Args:
            state: AgentState with emotion from webcam/classifier
        
        Returns:
            Updated AgentState with:
                - agent_response: Text for Furhat to speak
                - speech_duration: Estimated speaking time
                - continue_conversation: Whether to continue loop
        
        Example (Visual Loop):
            # In your main loop:
            while continue_loop:
                # 1. Capture frame and classify emotion
                emotion, confidence = classifier.predict(frame)
                
                # 2. Generate response
                result = graph.invoke({
                    "state": session,
                    "emotion": emotion,
                    "confidence": confidence,
                    "continue_conversation": True,
                    "speech_duration": 0.0
                })
                
                # 3. Send to Furhat
                furhat.say(result["state"].agent_response)
                
                # 4. Wait for speech to complete
                time.sleep(result["speech_duration"])
                
                # 5. Check if loop should continue
                continue_loop = result["continue_conversation"]
        """
        round_num = state["state"].conversation_round + 1
        emotion_name = state["emotion"].emotion_type.name if state["emotion"] else "NONE"
        confidence = state.get("confidence", 0.0)
        
        logger.info(f"▶️ Round {round_num} | Emotion: {emotion_name} ({confidence:.0%})")
        return self.graph.invoke(state)
