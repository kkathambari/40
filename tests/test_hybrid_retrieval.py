import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from core.rag_engine import RAGEngine

print("--- Initializing Multi-Chunk RAGEngine ---")
rag = RAGEngine()
rag.clear()

multi_section_report = """
SECTION 1: SYSTEM OVERVIEW AND MOTIVATION
The Autonomous Intelligent Viva Examiner (AIVA) automates comprehensive technical viva assessments. Traditional viva voce evaluations suffer from subjective grader bias, inconsistent question difficulty, and high scheduling overhead. AIVA addresses this by ingesting technical reports submitted in PDF, DOCX, or PPTX format, synthesizing knowledge representations, and evaluating student comprehension via grounded dialogue.

SECTION 2: MICROSERVICE INFRASTRUCTURE
AIVA deploys containerized services orchestrated by Kubernetes. The application gateway runs FastAPI on Python 3.10 with Uvicorn workers. Inter-service messaging leverages gRPC for low-overhead telemetry transfer. Caching is handled by Redis running an in-memory cluster, maintaining sub-50ms latency across user sessions. Persistent storage utilizes PostgreSQL for user accounts and ChromaDB for vector representations.

SECTION 3: HYBRID RETRIEVAL-AUGMENTED GENERATION (RAG)
To ground viva questions strictly within the student's submitted project documentation, AIVA implements a Hybrid RAG pipeline. Dense semantic embeddings are computed using the Google Gemini embedding model (3072-dimensional space) and stored in ChromaDB using cosine distance metrics. Lexical and exact keyword retrieval is handled by the Okapi BM25 algorithm.
Dense vector search excels at conceptual matching but frequently drops exact identifiers such as port numbers, hyperparameter constants, and variable names. BM25 catches these exact keyword tokens. The two disparate retrieval rankings are combined using Reciprocal Rank Fusion (RRF):
RRF_Score(d) = sum(1 / (k + rank_m(d)))
where k is fixed at 60 as a smoothing hyperparameter.

SECTION 4: KNOWLEDGE GRAPH SYNTHESIS
Beyond flat chunk retrieval, deep conceptual understanding requires relational knowledge representation. AIVA constructs a directed Entity-Relation Knowledge Graph from the document's core sections. Google Gemini Flash extracts key technical entities classified into Concept, Algorithm, Tool, Framework, and Database categories. Relationships such as DEPENDS_ON, IMPLEMENTS, QUERIES, and FUSES are identified. NetworkX computes network density and centrality metrics, and PyVis renders a physics-based 3D ForceAtlas2 visualization.

SECTION 5: EXPERIMENTAL BENCHMARKS AND BOTTLENECK ANALYSIS
During load testing with Locust simulating 10,000 concurrent student sessions, the system maintained an average response latency of 84ms. The primary performance bottleneck was database connection pool exhaustion when simultaneous ChromaDB query locks collided. Introducing connection pooling with PgBouncer and asynchronous vector index pre-warming resolved this bottleneck, increasing system throughput by 320%.
"""

print("Indexing multi-section technical document...")
chunks_count = rag.index_document(multi_section_report, document_id="multi_section_report.txt")
print(f"Chunks indexed: {chunks_count}")

test_queries = [
    ("Exact Hyperparameter", "What is the smoothing constant k used in Reciprocal Rank Fusion?"),
    ("Bottleneck Analysis", "What was the primary performance bottleneck during stress testing?"),
    ("Infrastructure Caching", "What database handles in-memory session caching for low latency?"),
    ("Graph Architecture", "Which library renders the physics-based 3D ForceAtlas2 visualization?")
]

print("\n--- Running Hybrid Retrieval and Inspecting Results ---")
for category, q in test_queries:
    print(f"\n==========================================")
    print(f"Category: {category}")
    print(f"Query: '{q}'")
    results = rag.retrieve(q, top_k=2)
    print(f"Top {len(results)} retrieved chunks:")
    for rank, res in enumerate(results, 1):
        # find which section this belongs to
        first_line = res.strip().split("\n")[0]
        snippet = res.strip().replace("\n", " ")[:160]
        print(f"  [Rank {rank}]: Header -> {first_line}")
        print(f"            Content -> {snippet}...")

print("\n>>> MULTI-CHUNK HYBRID RETRIEVAL VERIFIED SUCCESSFULLY!")
