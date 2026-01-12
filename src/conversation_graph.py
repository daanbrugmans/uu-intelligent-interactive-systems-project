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
    EMOTION_PROMPTS = {
        EmotionType.HAPPY: "enthusiastic, cheerful, and energetic. Match their positive energy.",
        EmotionType.NEUTRAL: "calm, professional, and helpful. Be friendly but measured.",
        EmotionType.SAD: "compassionate, gentle, and supportive. Show empathy and patience."
    }
    
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.7,
            max_output_tokens=100
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
            {
                "positive": "respond_positive",
                "negative": "respond_negative"
            }
        )
        
        # Convergence: both response paths lead to finalize
        workflow.add_edge("respond_positive", "finalize")
        workflow.add_edge("respond_negative", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _route_by_emotion(self, state: AgentState) -> Literal["positive", "negative"]:
        """Route to different response nodes based on emotion valence"""
        emotion = state.get("emotion")
        if emotion and emotion.emotion_type == EmotionType.HAPPY:
            return "positive"
        # SAD and NEUTRAL both route to negative (more careful/empathetic)
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
            stage = "greeting the customer"
        elif round_num <= 4:
            stage = "helping with their purchase"
        elif round_num <= 7:
            stage = "handling payment/checkout"
        else:
            stage = "saying goodbye"
        
        # Build prompt
        emotion_style = self.EMOTION_PROMPTS.get(emotion.emotion_type, "professional")
        
        system_prompt = f"""You are a friendly store cashier. Be {tone} and {emotion_style}.
Current stage: {stage}
Keep responses natural and brief (1-2 sentences)."""
        
        # Format recent history
        history_context = ""
        if state["state"].conversation_history:
            recent = state["state"].conversation_history[-2:]
            for turn in recent:
                history_context += f"\nCustomer: {turn.user_input}\nYou: {turn.agent_response}"
        
        if history_context:
            system_prompt += f"\n\nRecent conversation:{history_context}"
        
        # Call Gemini via LangChain
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ]
        
        try:
            response = self.llm.invoke(messages)
            state["state"].agent_response = response.content.strip()
            logger.info(f"Generated response: {state['state'].agent_response[:50]}...")
        except Exception as e:
            logger.error(f"LLM error: {e}")
            state["state"].agent_response = self._fallback_response(emotion.emotion_type)
        
        return state
    
    def _fallback_response(self, emotion_type: EmotionType) -> str:
        """Fallback responses when LLM fails"""
        fallbacks = {
            EmotionType.HAPPY: "Great! How can I help you today?",
            EmotionType.NEUTRAL: "Sure, let me know if you need anything.",
            EmotionType.SAD: "I'm here to help. Take your time."
        }
        return fallbacks.get(emotion_type, "How can I assist you?")
    
    def _finalize(self, state: AgentState) -> AgentState:
        """Record turn and check termination"""
        state["state"].conversation_round += 1
        
        if state["emotion"]:
            state["state"].add_turn(
                user_input=state["state"].user_input,
                agent_response=state["state"].agent_response,
                emotion=state["emotion"]
            )
        
        state["continue_conversation"] = not state["state"].should_terminate(120, 10)
        return state
    
    def invoke(self, state: AgentState) -> AgentState:
        """Execute conversation graph for one turn"""
        return self.graph.invoke(state)
