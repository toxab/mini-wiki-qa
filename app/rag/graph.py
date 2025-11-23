"""LangGraph RAG implementation"""
import logging
from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, END
from rag.retrieval import get_retriever
from rag.reranker import get_reranker
from rag.generation import get_generator
from rag.safety import get_pii_scrubber, get_injection_guard

logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    """State for RAG graph"""
    query: str
    chunks: List[Dict]
    answer: str
    use_rerank: bool
    metadata: Dict
    is_safe: bool  # Added for safety
    error: str  # Added for error handling


def injection_guard_node(state: RAGState) -> RAGState:
    """
    Check query for injection attempts

    Args:
        state: Current RAG state

    Returns:
        Updated state with safety check
    """
    logger.info("Injection Guard node: checking query...")

    guard = get_injection_guard()
    result = guard.check(state["query"])

    state["is_safe"] = result["is_safe"]
    state["metadata"] = {
        **state.get("metadata", {}),
        "injection_check": {
            "is_safe": result["is_safe"],
            "risk_level": result["risk_level"]
        }
    }

    if not result["is_safe"]:
        logger.warning(f"Injection detected: {result['detected_patterns']}")
        state["error"] = "Query blocked: potential injection attempt detected"
        state["answer"] = "⚠️ Your query was blocked for security reasons. Please rephrase your question."

    return state


def retrieve_node(state: RAGState) -> RAGState:
    """Retrieve relevant chunks"""
    # Skip if blocked by injection guard
    if not state.get("is_safe", True):
        logger.info("Retrieve node: skipped (query blocked)")
        return state

    logger.info(f"Retrieve node: query={state['query'][:50]}...")

    retriever = get_retriever()
    top_k = 20 if state.get("use_rerank", False) else 5

    chunks = retriever.retrieve(state["query"], top_k=top_k)

    state["chunks"] = chunks
    state["metadata"] = {
        **state.get("metadata", {}),
        "retrieval_count": len(chunks)
    }

    logger.info(f"Retrieved {len(chunks)} chunks")
    return state


def rerank_node(state: RAGState) -> RAGState:
    """Rerank chunks (conditional)"""
    # Skip if blocked
    if not state.get("is_safe", True):
        logger.info("Rerank node: skipped (query blocked)")
        return state

    if not state.get("use_rerank", False):
        logger.info("Rerank node: skipping (use_rerank=False)")
        return state

    logger.info("Rerank node: reranking chunks...")

    reranker = get_reranker()
    chunks = reranker.rerank(state["query"], state["chunks"], top_k=5)

    state["chunks"] = chunks
    state["metadata"] = {
        **state.get("metadata", {}),
        "reranked": True
    }

    logger.info(f"Reranked to {len(chunks)} chunks")
    return state


def generate_node(state: RAGState) -> RAGState:
    """Generate answer from chunks"""
    # Skip if blocked
    if not state.get("is_safe", True):
        logger.info("Generate node: skipped (query blocked)")
        return state

    logger.info("Generate node: generating answer...")

    generator = get_generator()
    answer = generator.generate(state["query"], state["chunks"])

    state["answer"] = answer
    logger.info(f"Generated answer: {answer[:100]}...")

    return state


def pii_scrubber_node(state: RAGState) -> RAGState:
    """
    Scrub PII from answer

    Args:
        state: Current RAG state

    Returns:
        Updated state with scrubbed answer
    """
    # Skip if no answer (blocked query)
    if not state.get("answer"):
        logger.info("PII Scrubber node: skipped (no answer)")
        return state

    logger.info("PII Scrubber node: checking for PII...")

    scrubber = get_pii_scrubber()
    result = scrubber.scrub(state["answer"])

    state["answer"] = result["text"]
    state["metadata"] = {
        **state.get("metadata", {}),
        "pii_scrubbed": {
            "was_scrubbed": result["was_scrubbed"],
            "pii_types": result["pii_detected"]
        }
    }

    if result["was_scrubbed"]:
        logger.warning(f"PII scrubbed from answer: {result['pii_detected']}")

    return state


def create_rag_graph() -> StateGraph:
    """
    Create RAG graph with safety layers

    Returns:
        Compiled LangGraph
    """
    # Create graph
    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("injection_guard", injection_guard_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("pii_scrubber", pii_scrubber_node)

    # Add edges (safety first!)
    workflow.set_entry_point("injection_guard")
    workflow.add_edge("injection_guard", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", "pii_scrubber")
    workflow.add_edge("pii_scrubber", END)

    # Compile
    graph = workflow.compile()

    logger.info("RAG graph created with safety layers")
    return graph


# Global graph instance
_rag_graph = None

def get_rag_graph() -> StateGraph:
    """Get or create RAG graph"""
    global _rag_graph
    if _rag_graph is None:
        _rag_graph = create_rag_graph()
    return _rag_graph