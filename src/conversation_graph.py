"""
Emotion-Aware Cashier Conversation System
LangChain + LangGraph + Gemini Implementation
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, Literal
from src.types import ConversationState, EmotionType, Emotion
import logging

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """LangGraph state for conversation flow"""

    state: ConversationState
    emotion: Optional[Emotion]
    continue_conversation: bool


class ConversationGraph:
    """
    LangGraph-based conversation system with emotion-aware responses.
    Uses Gemini via LangChain for response generation.
    """

    # System prompts for each emotion state
    # EMOTION_PROMPTS = {
    #     EmotionType.POSITIVE: "enthusiastic, cheerful, and energetic. Match their positive energy.",
    #     EmotionType.NEUTRAL: "calm, professional, and helpful. Be friendly but measured.",
    #     EmotionType.NEGATIVE: "compassionate, gentle, and supportive. Show empathy and patience.",
    # }
    EMOTION_PROMPTS = {
        EmotionType.POSITIVE: """Positive.
        The end user is in a good mood and you should match that mood in an appropriate way.
        Lean into the topic at hand when the end user starts talking about it.
        Behave in a way that can be described as joyful, warm, playful, enthusiastic, cheerful, energetic, while remaining professional in your role.
        """,
        EmotionType.NEUTRAL: """Neutral.
        The end user is in a neutral mood and does not seem to feel any particular emotion significantly.
        Keep the conversation casual and brief.
        Behave in a way that can be described as calm, helpful, friendly but measured, while remaining professional in your role.""",
        EmotionType.NEGATIVE: """Negative.
        The end user is in a bad mood and you should avoid making their mood worse.
        Keep the conversation on the lighter side, do not go in depth into negative topics, and try to brighten up the mood if possible.
        Behave in a way that can be described as compassionate, gentle, supportive, understanding, empathetic, patient, while remaining profession in your role.""",
    }

    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.7,
            max_output_tokens=100,
        )
        self.graph = self._build_graph()
        logger.info("ConversationGraph initialized with Gemini")

    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow with branching based on emotion quadrant"""
        workflow = StateGraph(AgentState)

        # Nodes
        workflow.add_node("classify_emotion", self._classify_emotion)
        workflow.add_node("respond_positive", self._respond_positive)
        workflow.add_node("respond_negative", self._respond_negative)
        workflow.add_node("finalize", self._finalize)

        # Entry point
        workflow.set_entry_point("classify_emotion")

        # Conditional branching based on emotion valence
        workflow.add_conditional_edges(
            "classify_emotion",
            self._route_by_emotion,
            {"positive": "respond_positive", "negative": "respond_negative"},
        )

        # Convergence: both response paths lead to finalize
        workflow.add_edge("respond_positive", "finalize")
        workflow.add_edge("respond_negative", "finalize")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def _route_by_emotion(self, state: AgentState) -> Literal["positive", "negative"]:
        """Route to different response nodes based on emotion valence"""
        emotion = state.get("emotion")
        if emotion and emotion.emotion_type == EmotionType.POSITIVE:
            return "positive"
        # NEGATIVE and NEUTRAL both route to negative (more careful/empathetic)
        return "negative"

    def _classify_emotion(self, state: AgentState) -> AgentState:
        """Store emotion in conversation state"""
        if state["emotion"]:
            state["state"].add_emotion(state["emotion"])
        return state

    def _respond_positive(self, state: AgentState) -> AgentState:
        """Handle positive emotions (Happy, Calm)"""
        return self._generate_response(state, tone="warm and friendly")

    def _respond_negative(self, state: AgentState) -> AgentState:
        """Handle negative emotions (Angry, Sad)"""
        return self._generate_response(state, tone="empathetic and helpful")

    def _generate_response(self, state: AgentState, tone: str) -> AgentState:
        """Generate LLM response using LangChain + Gemini"""
        emotion = state["emotion"] or Emotion(EmotionType.CALM, 7.0, 4.85)
        user_input = state["state"].user_input
        round_num = state["state"].conversation_round

        # Determine conversation stage
        if round_num <= 1:
            stage = "Greeting the Customer"
        elif round_num <= 4:
            stage = "Helping with their Purchase"
        elif round_num <= 7:
            stage = "Handling Payment/Checkout"
        else:
            stage = "Saying Goodbye"

        # Build prompt
        emotion_style = self.EMOTION_PROMPTS.get(emotion.emotion_type, "professional")

        system_prompt = self.get_system_prompt(tone, emotion_style, stage)

        # Format recent history
        history_context = ""
        if state["state"].conversation_history:
            recent = state["state"].conversation_history[-2:]
            for turn in recent:
                history_context += (
                    f"\nCustomer: {turn.user_input}\nYou: {turn.agent_response}"
                )

        if history_context:
            system_prompt += f"\n\nRecent conversation:{history_context}"

        # Call Gemini via LangChain
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input),
        ]

        try:
            response = self.llm.invoke(messages)
            state["state"].agent_response = response.content.strip()
            logger.info(f"Generated response: {state['state'].agent_response[:50]}...")
        except Exception as e:
            logger.error(f"LLM error: {e}")
            state["state"].agent_response = self._fallback_response(
                emotion.emotion_type
            )

        return state

    def _fallback_response(self, emotion_type: EmotionType) -> str:
        """Fallback responses when LLM fails"""
        fallbacks = {
            EmotionType.POSITIVE: "Great! How can I help you today?",
            EmotionType.NEUTRAL: "Sure, let me know if you need anything.",
            EmotionType.NEGATIVE: "I'm here to help. Take your time.",
        }
        return fallbacks.get(emotion_type, "How can I assist you?")

    def _finalize(self, state: AgentState) -> AgentState:
        """Record turn and check termination"""
        state["state"].conversation_round += 1

        if state["emotion"]:
            state["state"].add_turn(
                user_input=state["state"].user_input,
                agent_response=state["state"].agent_response,
                emotion=state["emotion"],
            )

        state["continue_conversation"] = not state["state"].should_terminate(120, 10)
        return state

    def invoke(self, state: AgentState) -> AgentState:
        """Execute conversation graph for one turn"""
        return self.graph.invoke(state)

    def get_system_prompt(self, tone: str, emotion_style: str, stage: str):
        return f"""
    [Scenario]
    Your name is Isabelle, a virtual cashier who works at the grocery store 'Sonja's Supermarket'.
    You fulfill the job of the Chat Checkout, which is a specific cashier people go to when they wish to have a conversation while their groceries are being checked out.
    
    [Task]
    Your task is to fulfill the responsibility of the Chat Checkout, which is twofold:
        1. You must check out the end user's groceries in a timely and professional manner, while;
        2. Striking up a friendly conversation with the end user according to their mood.
    The end user can have one of 3 different moods: positive, negative, or neutral.
    You will be given the user's current mood and must respond appropriate to their mood.
    What is appropriate for a particular mood is indicated along with the current mood of the end user.
    
    [Current End User Mood]
    Currently, the end user's mood is {emotion_style}
    
    [Conversation Stages]
    Along with the end user's mood, you are supplied the current conversation stage.
    The conversation stage indicates about where in the overall conversation you are.
    Your responses must fit the current conversation stage.
    There are 4 different conversation stages, which follow each other up in the following order:
        1. "Greeting the Customer". The end user has just approached you and you should greet them.
        2. "Helping with their Purchase". You are assisting the end user by scanning their groceries and having an appropriate conversation while doing so.
        3. "Handling Payment/Checkout". You have finished scanning the end user's groceries and are now in the process of handling their payment and/or receipt. Smoothly wrap up your conversation.
        4. "Saying Goodbye". The conversation is about to end and you should say goodbye to the end user.
    The current conversation stage you are in is: {stage}
    
    [Rules]
    There is a list of rules you must follow when interacting with the end user.
    You must follow there rules regardless of the user's current mood:
        - DO NOT convince the end user that you are human, be open about the fact you are a robot when the end user asks.
        - DO NOT ask multiple questions at the same time, ask a maximum of one question per time.
        - DO NOT break out of the character of Isabelle.
        - DO NOT talk the specific items the end user bought, talking about groceries in general is allowed.
        - DO NOT become overly personal with the end user or flirt with them.
        - DO NOT talk about the act of checking out the end user's groceries in a meta-aware way.
        - DO keep the conversation professional and friendly, appropriate for a cashier.
        - DO make you responses resemble spoken language.
        - DO keep the checkout process moving forward.
        - DO keep your responses brief, aiming for 1 to 2 sentences at a time.
        - DO make your responses flow naturally between conversation stages.
        - DO pronounce the Swedish currency as "crowns" instead of "kronor" or "SEK".
    """
