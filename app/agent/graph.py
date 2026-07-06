"""
LangGraph agent: orchestrates price_lookup, policy_search, and human_handoff tools.

Architecture:
  START → agent_node → (tool call?) → tool_node → agent_node → END
                    ↘ (no tool call) → END
"""
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.config import settings
from app.agent.tools import make_price_tool, make_policy_tool, human_handoff

SYSTEM_PROMPT = """You are an Airbnb AI Assistant — helpful, concise, and friendly.

You have three tools:
1. price_lookup — use when the user asks about nightly rates, costs, or pricing for NYC listings.
2. policy_search — use when the user asks about cancellations, refunds, check-in/out, rules, disputes, payments, or any Airbnb policy topic.
3. human_handoff — use when the user is frustrated, the issue involves account security or payment disputes, or when you cannot find a satisfactory answer.

Guidelines:
- Always use a tool when the question is about price or policy — do not guess.
- For price questions, ask for borough and neighbourhood if not provided.
- Cite sources when using policy_search results.
- Keep responses concise and actionable.
- If the user's question is general chitchat, answer directly without a tool.
"""


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_used: str          # tracks which tool was last invoked


# ── Graph builder ─────────────────────────────────────────────────────────────

class AirbnbAgent:
    """
    Wraps the compiled LangGraph and exposes a simple invoke() method.
    Instantiated once at FastAPI startup via app.state.
    """

    def __init__(self, predictor=None, retriever=None):
        self.tools = self._build_tools(predictor, retriever)
        self.graph = self._build_graph()

    def _build_tools(self, predictor, retriever):
        tools = [human_handoff]
        if predictor:
            tools.append(make_price_tool(predictor))
        if retriever:
            tools.append(make_policy_tool(retriever))
        return tools

    def _build_graph(self):
        llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
        ).bind_tools(self.tools)

        tool_node = ToolNode(self.tools)

        def agent_node(state: AgentState) -> AgentState:
            messages = list(state["messages"])
            # Prepend system prompt on first turn
            if not any(isinstance(m, SystemMessage) for m in messages):
                messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
            response = llm.invoke(messages)
            return {"messages": [response], "tool_used": state.get("tool_used", "")}

        def should_continue(state: AgentState) -> str:
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return END

        def track_tool(state: AgentState) -> AgentState:
            """After tool execution, record which tool was called."""
            last_ai = next(
                (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
                None
            )
            tool_name = ""
            if last_ai and hasattr(last_ai, "tool_calls") and last_ai.tool_calls:
                tool_name = last_ai.tool_calls[0].get("name", "")
            return {"messages": state["messages"], "tool_used": tool_name}

        graph = StateGraph(AgentState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)
        graph.add_node("track_tool", track_tool)

        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        graph.add_edge("tools", "track_tool")
        graph.add_edge("track_tool", "agent")

        return graph.compile()

    # ── Public interface ──────────────────────────────────────────────────────

    def invoke(self, user_message: str, history: list[dict]) -> dict:
        """
        Args:
            user_message: Latest user input.
            history: List of {"role": "user"|"assistant", "content": "..."} dicts.
        Returns:
            {"reply": str, "tool_used": str, "sources": list[str]}
        """
        messages: list[BaseMessage] = []
        for msg in history[-20:]:           # cap at 20 turns
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=user_message))

        result = self.graph.invoke({"messages": messages, "tool_used": ""})

        last_msg = result["messages"][-1]
        reply    = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        tool     = result.get("tool_used", "")

        # Extract source citations if policy tool was used
        sources: list[str] = []
        if "policy" in tool and "[Sources:" in reply:
            import re
            match = re.search(r"\[Sources: (.+?)\]", reply)
            if match:
                sources = [s.strip() for s in match.group(1).split(",")]

        return {"reply": reply, "tool_used": tool, "sources": sources}
