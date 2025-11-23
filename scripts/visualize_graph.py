"""Visualize LangGraph RAG pipeline"""
import sys
sys.path.insert(0, '/app')

from rag.graph import create_rag_graph

def main():
    """Generate graph visualization"""
    print("🎨 Creating RAG graph visualization...")

    # Create graph
    graph = create_rag_graph()
    graph_obj = graph.get_graph()

    # Manual beautiful ASCII diagram
    ascii_art = """
╔════════════════════════════════════════════════════════════╗
║                   RAG PIPELINE FLOW                        ║
╚════════════════════════════════════════════════════════════╝

                    ┌──────────────────┐
                    │   User Query     │
                    └────────┬─────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │   injection_guard      │  🛡️  Security Check
                │  (prompt injection)    │
                └────────────┬───────────┘
                             │
                    ┌────────┴────────┐
                    │   is_safe?      │
                    └────────┬────────┘
                          Yes│  No → Block
                             ▼
                ┌────────────────────────┐
                │      retrieve          │  🔍  Semantic Search
                │   (Qdrant vector DB)   │      (top_k chunks)
                └────────────┬───────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │       rerank           │  🎯  Cross-Encoder
                │  (optional, if flag)   │      (improve order)
                └────────────┬───────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │      generate          │  🤖  LLM Generation
                │   (answer from docs)   │      (with context)
                └────────────┬───────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │    pii_scrubber        │  🔒  Privacy Filter
                │  (remove PII data)     │      (emails, phones)
                └────────────┬───────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ Final Answer   │
                    └────────────────┘

╔════════════════════════════════════════════════════════════╗
║  STATE FLOW:                                               ║
║  • query: str                                              ║
║  • chunks: List[Dict]                                      ║
║  • answer: str                                             ║
║  • use_rerank: bool                                        ║
║  • is_safe: bool                                           ║
║  • metadata: Dict                                          ║
╚════════════════════════════════════════════════════════════╝
"""

    # Node details
    nodes_info = """
╔════════════════════════════════════════════════════════════╗
║                     NODES DETAILS                          ║
╚════════════════════════════════════════════════════════════╝

1. 🛡️  injection_guard
   ├─ Input: query
   ├─ Action: Check for prompt injection patterns
   ├─ Output: is_safe flag
   └─ Blocks: "ignore instructions", "system:", etc.

2. 🔍  retrieve
   ├─ Input: query
   ├─ Action: Semantic search in Qdrant
   ├─ Model: sentence-transformers/all-MiniLM-L6-v2
   └─ Output: top_k chunks (5 or 20)

3. 🎯  rerank (conditional)
   ├─ Input: chunks
   ├─ Action: Cross-encoder scoring
   ├─ Model: cross-encoder/ms-marco-MiniLM-L-6-v2
   └─ Output: reranked top-5 chunks

4. 🤖  generate
   ├─ Input: query + chunks
   ├─ Action: LLM answer generation
   ├─ Model: phi-3-mini (via LM Studio)
   └─ Output: answer text

5. 🔒  pii_scrubber
   ├─ Input: answer
   ├─ Action: Remove PII (emails, phones, SSN, CC)
   ├─ Patterns: regex-based detection
   └─ Output: cleaned answer

╔════════════════════════════════════════════════════════════╗
║  METADATA TRACKING:                                        ║
║  • injection_check: {is_safe, risk_level}                 ║
║  • retrieval_count: number of chunks                       ║
║  • reranked: boolean flag                                  ║
║  • pii_scrubbed: {was_scrubbed, pii_types}                ║
╚════════════════════════════════════════════════════════════╝
"""

    # Print to console
    print(ascii_art)
    print(nodes_info)

    # Save to file
    output_path = "/data/rag_pipeline_visualization.txt"
    with open(output_path, "w") as f:
        f.write(ascii_art)
        f.write("\n\n")
        f.write(nodes_info)
        f.write("\n\n")
        f.write("="*60 + "\n")
        f.write(f"Graph nodes: {list(graph_obj.nodes.keys())}\n")
        f.write(f"Graph edges: {[(e[0], e[1]) for e in graph_obj.edges]}\n")

    print(f"\n✅ Visualization saved to: {output_path}")
    print("📊 Pipeline visualization complete!")

if __name__ == "__main__":
    main()