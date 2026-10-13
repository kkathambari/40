import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from dotenv import load_dotenv
load_dotenv()

from core.knowledge_graph import KGBuilder

sample_text = """
The Autonomous Intelligent Viva Examiner (AIVA) automates comprehensive technical viva assessments.
AIVA employs FastAPI as its backend web framework communicating via RESTful endpoints.
ChromaDB serves as the localized vector database, paired with Google Gemini embeddings for dense semantic retrieval.
Okapi BM25 provides sparse keyword indexing, which is combined with dense search using Reciprocal Rank Fusion (RRF k=60).
Redis handles in-memory session caching to sustain high throughput and reduce latency below 50ms.
Docker packages the microservice containers, and Kubernetes orchestrates auto-scaling across GPU nodes.
The Knowledge Graph module uses Gemini Flash to extract technical entities and relationships, which NetworkX structures and PyVis renders with ForceAtlas2 physics simulation.
"""

print("--- Initializing KGBuilder ---")
kg = KGBuilder()

print("Extracting entities and relationships via Gemini Flash...")
extraction = kg.extract_graph_from_text(sample_text)

if not extraction:
    print("FAILED: Extraction returned None.")
    exit(1)

print(f"\n[Extraction Succeeded in {kg.generation_time}s]")
print(f"Total Nodes Extracted: {len(extraction.nodes)}")
print(f"Total Relationships Extracted: {len(extraction.relationships)}")

print("\n--- Extracted Sample Nodes ---")
for node in extraction.nodes[:8]:
    print(f"  • Node: {node.id:20} | Type: {node.type}")

print("\n--- Extracted Sample Relationships ---")
for rel in extraction.relationships[:8]:
    print(f"  • {rel.source} --[{rel.type}]--> {rel.target}")

print("\n--- Building NetworkX Graph & Metadata ---")
kg.build_networkx_graph(extraction)

if os.path.exists("graph_metadata.json"):
    with open("graph_metadata.json", "r") as f:
        meta = json.load(f)
        print("Graph Metadata:", json.dumps(meta, indent=2))

out_html = kg.generate_html_visualization("kg_test_visualization.html")
print(f"Generated Visualization: {out_html} (Size: {os.path.getsize(out_html)} bytes)")

print("\n>>> GEMINI ENTITY & RELATIONSHIP EXTRACTION TEST PASSED!")
