"""LangGraph RAG implementation"""
import logging
from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, END
from rag.retrieval import get_retriever
from rag.reranker import get_reranker
from rag.generation import get_generator

logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    """State for RAG graph"""
    query: str
    chunks: List[Dict]
    answer: str
    use_rerank: bool
    metadata: Dict


def retrieve_node(state: RAGState) -> RAGState:
    """
    Retrieve relevant chunks

    Args:
        state: Current RAG state

    Returns:
        Updated state with chunks
    """
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
    """
    Rerank chunks (conditional node)

    Args:
        state: Current RAG state

    Returns:
        Updated state with reranked chunks
    """
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
    """
    Generate answer from chunks

    Args:
        state: Current RAG state

    Returns:
        Updated state with answer
    """
    logger.info("Generate node: generating answer...")

    generator = get_generator()
    answer = generator.generate(state["query"], state["chunks"])

    state["answer"] = answer
    logger.info(f"Generated answer: {answer[:100]}...")

    return state


def create_rag_graph() -> StateGraph:
    """
    Create RAG graph

    Returns:
        Compiled LangGraph
    """
    # Create graph
    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate", generate_node)

    # Add edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", END)

    # Compile
    graph = workflow.compile()

    logger.info("RAG graph created successfully")
    return graph


# Global graph instance
_rag_graph = None


def get_rag_graph() -> StateGraph:
    """Get or create RAG graph"""
    global _rag_graph
    if _rag_graph is None:
        _rag_graph = create_rag_graph()
    return _rag_graph