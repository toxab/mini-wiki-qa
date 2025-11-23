"""Generate PNG image of RAG pipeline using Graphviz"""
import sys

sys.path.insert(0, '/app')


def main():
    """Generate pipeline visualization as PNG"""
    print("🎨 Generating RAG pipeline PNG...")

    try:
        import pygraphviz as pgv

        # Create directed graph
        G = pgv.AGraph(directed=True, strict=True, rankdir='TB')

        # Set graph attributes
        G.graph_attr.update({
            'dpi': '150',
            'bgcolor': 'white',
            'fontname': 'Arial',
            'fontsize': '12',
            'label': 'RAG Pipeline with Safety Layers',
            'labelloc': 't',
            'pad': '0.5'
        })

        # Node style
        node_style = {
            'shape': 'box',
            'style': 'rounded,filled',
            'fontname': 'Arial',
            'fontsize': '11',
            'width': '2.5',
            'height': '0.8'
        }

        # Add nodes with colors
        G.add_node('start', label='User Query', fillcolor='#e1f5e1', **node_style)
        G.add_node('injection_guard', label='🛡️ Injection Guard\n(Security Check)', fillcolor='#ffe6e6', **node_style)
        G.add_node('retrieve', label='🔍 Retrieve\n(Semantic Search)', fillcolor='#e6f3ff', **node_style)
        G.add_node('rerank', label='🎯 Rerank\n(Cross-Encoder)', fillcolor='#fff5e6', **node_style)
        G.add_node('generate', label='🤖 Generate\n(LLM Answer)', fillcolor='#f0e6ff', **node_style)
        G.add_node('pii_scrubber', label='🔒 PII Scrubber\n(Privacy Filter)', fillcolor='#ffe6f0', **node_style)
        G.add_node('end', label='Final Answer', fillcolor='#e1f5e1', **node_style)

        # Add edges
        G.add_edge('start', 'injection_guard', color='#333333', penwidth='2')
        G.add_edge('injection_guard', 'retrieve', color='#4CAF50', penwidth='2', label='is_safe=True')
        G.add_edge('retrieve', 'rerank', color='#2196F3', penwidth='2')
        G.add_edge('rerank', 'generate', color='#FF9800', penwidth='2')
        G.add_edge('generate', 'pii_scrubber', color='#9C27B0', penwidth='2')
        G.add_edge('pii_scrubber', 'end', color='#333333', penwidth='2')

        # Add blocked path
        G.add_edge('injection_guard', 'end', color='#F44336', penwidth='2',
                   label='is_safe=False\n(blocked)', style='dashed')

        # Save as PNG
        output_path = '/data/rag_pipeline.png'
        G.draw(output_path, prog='dot')

        print(f"✅ PNG saved to: {output_path}")
        print("📊 Pipeline image generated successfully!")

        # Also save as SVG (scalable)
        svg_path = '/data/rag_pipeline.svg'
        G.draw(svg_path, prog='dot')
        print(f"✅ SVG saved to: {svg_path}")

    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Make sure pygraphviz is installed!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()