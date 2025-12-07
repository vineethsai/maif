"""
Pre-built LangGraph patterns with MAIF provenance.

These patterns provide ready-to-use graph configurations for common
use cases, all with built-in cryptographic provenance tracking.

Usage:
    from maif.integrations.langgraph.patterns import create_chat_graph
    
    app = create_chat_graph("chat_history.maif", llm=my_llm)
    result = app.invoke({"messages": [...]}, config)
"""

from typing import (
    Any, Callable, Dict, List, Optional, TypedDict, Annotated, Sequence
)
from operator import add
from pathlib import Path

from .checkpointer import MAIFCheckpointer


# =============================================================================
# State Definitions
# =============================================================================

class ChatMessage(TypedDict):
    """A single chat message."""
    role: str  # "user", "assistant", "system"
    content: str


class ChatState(TypedDict):
    """State for chat applications."""
    messages: Annotated[List[ChatMessage], add]


class RAGState(TypedDict):
    """State for RAG applications."""
    messages: Annotated[List[ChatMessage], add]
    query: str
    context: List[str]
    citations: List[Dict[str, Any]]


class MultiAgentState(TypedDict):
    """State for multi-agent applications."""
    messages: Annotated[List[ChatMessage], add]
    current_agent: str
    agent_outputs: Dict[str, Any]
    task_complete: bool


# =============================================================================
# Pattern: Simple Chat with Memory
# =============================================================================

def create_chat_graph(
    artifact_path: str,
    llm_func: Callable[[List[ChatMessage]], str],
    *,
    system_prompt: Optional[str] = None,
    max_history: int = 20,
    agent_id: str = "chat_agent",
):
    """Create a simple chat graph with MAIF provenance.
    
    This creates a chat application that:
    - Maintains conversation history
    - Truncates old messages to stay within limits
    - Logs all interactions to a MAIF artifact
    
    Args:
        artifact_path: Path for the MAIF artifact
        llm_func: Function that takes messages and returns response string
        system_prompt: Optional system prompt to prepend
        max_history: Maximum messages to keep in history
        agent_id: Agent identifier for provenance
        
    Returns:
        Compiled LangGraph application
        
    Example:
        def my_llm(messages):
            # Your LLM logic here
            return "Hello!"
        
        app = create_chat_graph("chat.maif", my_llm)
        result = app.invoke({
            "messages": [{"role": "user", "content": "Hi!"}]
        }, {"configurable": {"thread_id": "session-1"}})
    """
    from langgraph.graph import StateGraph, START, END
    
    def process_message(state: ChatState) -> ChatState:
        """Process user message and generate response."""
        messages = state["messages"]
        
        # Add system prompt if needed
        if system_prompt and (not messages or messages[0]["role"] != "system"):
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        # Truncate if needed
        if len(messages) > max_history:
            # Keep system prompt + recent messages
            if messages[0]["role"] == "system":
                messages = [messages[0]] + messages[-(max_history - 1):]
            else:
                messages = messages[-max_history:]
        
        # Generate response
        response = llm_func(messages)
        
        return {
            "messages": [{"role": "assistant", "content": response}]
        }
    
    # Build graph
    graph = StateGraph(ChatState)
    graph.add_node("respond", process_message)
    graph.add_edge(START, "respond")
    graph.add_edge("respond", END)
    
    # Compile with MAIF checkpointer
    checkpointer = MAIFCheckpointer(artifact_path, agent_id=agent_id)
    return graph.compile(checkpointer=checkpointer)


# =============================================================================
# Pattern: RAG with Citations
# =============================================================================

def create_rag_graph(
    artifact_path: str,
    retriever_func: Callable[[str], List[Dict[str, Any]]],
    llm_func: Callable[[str, List[str]], str],
    *,
    top_k: int = 5,
    agent_id: str = "rag_agent",
):
    """Create a RAG graph with citation tracking and MAIF provenance.
    
    This creates a RAG application that:
    - Retrieves relevant documents
    - Tracks citations for each response
    - Logs all retrieval and generation steps
    
    Args:
        artifact_path: Path for the MAIF artifact
        retriever_func: Function that takes query and returns documents
                       Each doc should have 'content' and 'source' keys
        llm_func: Function that takes (query, contexts) and returns response
        top_k: Number of documents to retrieve
        agent_id: Agent identifier for provenance
        
    Returns:
        Compiled LangGraph application
        
    Example:
        def my_retriever(query):
            return [{"content": "...", "source": "doc1.pdf"}]
        
        def my_llm(query, contexts):
            return "Based on the documents..."
        
        app = create_rag_graph("rag.maif", my_retriever, my_llm)
    """
    from langgraph.graph import StateGraph, START, END
    
    def retrieve(state: RAGState) -> RAGState:
        """Retrieve relevant documents."""
        query = state["query"]
        docs = retriever_func(query)[:top_k]
        
        contexts = [d.get("content", str(d)) for d in docs]
        citations = [
            {"source": d.get("source", "unknown"), "index": i}
            for i, d in enumerate(docs)
        ]
        
        return {
            "context": contexts,
            "citations": citations,
            "messages": [],
        }
    
    def generate(state: RAGState) -> RAGState:
        """Generate response with citations."""
        query = state["query"]
        contexts = state["context"]
        
        response = llm_func(query, contexts)
        
        return {
            "messages": [{"role": "assistant", "content": response}],
            "context": [],
            "citations": [],
        }
    
    # Build graph
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    
    # Compile with MAIF checkpointer
    checkpointer = MAIFCheckpointer(artifact_path, agent_id=agent_id)
    return graph.compile(checkpointer=checkpointer)


# =============================================================================
# Pattern: Multi-Agent Router
# =============================================================================

def create_multi_agent_graph(
    artifact_path: str,
    agents: Dict[str, Callable[[MultiAgentState], Dict[str, Any]]],
    router_func: Callable[[MultiAgentState], str],
    *,
    max_iterations: int = 10,
    agent_id: str = "multi_agent_orchestrator",
):
    """Create a multi-agent graph with routing and MAIF provenance.
    
    This creates a multi-agent application that:
    - Routes tasks to specialized agents
    - Tracks each agent's output
    - Logs all agent interactions for audit
    
    Args:
        artifact_path: Path for the MAIF artifact
        agents: Dict mapping agent names to their functions
        router_func: Function that determines next agent or "FINISH"
        max_iterations: Maximum agent calls before forcing completion
        agent_id: Agent identifier for provenance
        
    Returns:
        Compiled LangGraph application
        
    Example:
        def researcher(state):
            return {"research": "findings..."}
        
        def writer(state):
            return {"draft": "article..."}
        
        def router(state):
            if not state["agent_outputs"].get("research"):
                return "researcher"
            if not state["agent_outputs"].get("draft"):
                return "writer"
            return "FINISH"
        
        app = create_multi_agent_graph(
            "multi_agent.maif",
            {"researcher": researcher, "writer": writer},
            router
        )
    """
    from langgraph.graph import StateGraph, START, END
    
    iteration_count = {"count": 0}
    
    def route_to_agent(state: MultiAgentState) -> MultiAgentState:
        """Route to the next agent."""
        next_agent = router_func(state)
        
        iteration_count["count"] += 1
        if iteration_count["count"] >= max_iterations:
            next_agent = "FINISH"
        
        if next_agent == "FINISH":
            return {
                "current_agent": "FINISH",
                "task_complete": True,
                "messages": [],
                "agent_outputs": {},
            }
        
        return {
            "current_agent": next_agent,
            "task_complete": False,
            "messages": [],
            "agent_outputs": {},
        }
    
    def execute_agent(state: MultiAgentState) -> MultiAgentState:
        """Execute the current agent."""
        current = state["current_agent"]
        
        if current not in agents:
            return {
                "messages": [{"role": "system", "content": f"Unknown agent: {current}"}],
                "agent_outputs": {},
                "current_agent": "",
                "task_complete": False,
            }
        
        result = agents[current](state)
        
        return {
            "agent_outputs": {current: result},
            "messages": [{"role": "assistant", "content": f"[{current}] completed"}],
            "current_agent": "",
            "task_complete": False,
        }
    
    def should_continue(state: MultiAgentState) -> str:
        """Check if we should continue routing."""
        if state.get("task_complete", False):
            return "end"
        return "execute"
    
    # Build graph
    graph = StateGraph(MultiAgentState)
    graph.add_node("router", route_to_agent)
    graph.add_node("execute", execute_agent)
    
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        should_continue,
        {"execute": "execute", "end": END}
    )
    graph.add_edge("execute", "router")
    
    # Compile with MAIF checkpointer
    checkpointer = MAIFCheckpointer(artifact_path, agent_id=agent_id)
    return graph.compile(checkpointer=checkpointer)


# =============================================================================
# Utility Functions
# =============================================================================

def finalize_graph(app) -> None:
    """Finalize the MAIF artifact for a compiled graph.
    
    Call this when you're done using a graph to seal the artifact.
    
    Args:
        app: A compiled LangGraph application with MAIFCheckpointer
    """
    if hasattr(app, "checkpointer") and isinstance(app.checkpointer, MAIFCheckpointer):
        app.checkpointer.finalize()


def get_artifact_path(app) -> Optional[str]:
    """Get the MAIF artifact path for a compiled graph.
    
    Args:
        app: A compiled LangGraph application
        
    Returns:
        Path to the MAIF artifact, or None if not using MAIFCheckpointer
    """
    if hasattr(app, "checkpointer") and isinstance(app.checkpointer, MAIFCheckpointer):
        return app.checkpointer.get_artifact_path()
    return None

