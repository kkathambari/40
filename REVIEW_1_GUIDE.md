# Review 1 Presentation & Viva Defense Guide (40% Milestone)
## AIVA: Document Ingestion, Hybrid RAG & Knowledge Graph Engine

---

## 1. Milestone Focus & Objectives
By the 40% milestone, the system must prove that it can take any unformatted student technical report (**PDF, DOCX, PPTX**), parse its content, extract deep semantic relationships into a Knowledge Graph, and provide fast, grounded hybrid retrieval.

---

## 2. Core Technical Architecture

### A. Multi-Format Document Ingestion (`core/parser.py`)
*   **PDF Extraction:** Page-by-page streaming via `pdfplumber`.
*   **DOCX Extraction:** Hierarchical paragraph extraction via `python-docx`.
*   **PPTX Extraction:** Slide-by-slide shape and text-frame parsing via `python-pptx`.

### B. Hybrid RAG Engine (`core/rag_engine.py`)
*   **Chunking Strategy:** LangChain `RecursiveCharacterTextSplitter` with `chunk_size = 1000` and `chunk_overlap = 200` to preserve cross-boundary semantic context.
*   **Dense Semantic Search:** Google Gemini embeddings (`models/gemini-embedding-2`, 3072 dimensions) indexed in `ChromaDB` using cosine similarity (`hnsw:space: cosine`).
*   **Sparse Lexical Search:** `Okapi BM25` (`rank-bm25`) for exact keyword matches, variable names, and numerical constants.
*   **Reciprocal Rank Fusion (RRF):**
    $$\text{RRF\_Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
    Where:
    - $M$ is the set of rankers (Dense ChromaDB and Sparse BM25).
    - $r_m(d)$ is the 1-based rank position of chunk $d$ in system $m$.
    - $k = 60$ is the standard smoothing constant preventing individual high-rank outliers from dominating.

### C. Entity-Relation Knowledge Graph (`core/knowledge_graph.py`)
*   **LLM Extraction:** Google Gemini 2.5 Flash with structured Pydantic schema (`GraphExtraction(nodes, relationships)`).
*   **Node Categorization:** `Concept`, `Framework`, `Algorithm`, `Tool`, `Database`, `System`.
*   **Relationship Semantics:** `DEPENDS_ON`, `IMPLEMENTS`, `QUERIES`, `FUSES`, `AUTOMATES`, `CONTAINERIZED_BY`.
*   **Graph Metrics:** `NetworkX` calculates node count, edge count, concept taxonomy diversity, and directed graph density:
    $$\text{Density} = \frac{|E|}{|V|(|V|-1)}$$
*   **Interactive 3D Rendering:** `PyVis` with `ForceAtlas2` physics engine for dynamic node repulsion, gravity centering, and physics-based stabilization.

---

## 3. Reviewer Live Demonstration Steps

1.  **Run the automated CLI demo:**
    ```bash
    python demo_40.py
    ```
2.  **Or launch the FastAPI server:**
    ```bash
    python -m uvicorn main:app --port 8001 --reload
    ```
3.  **Execute the endpoints from Swagger (`http://127.0.0.1:8001/docs`):**
    - `POST /upload_and_build_kg`: Upload `sample_reports/AIVA_Technical_Report.pdf` (or `.docx` / `.pptx`).
    - `GET /query_rag?query=What is the rank constant used in Reciprocal Rank Fusion?`: Demonstrates fused RRF retrieval.
    - `GET /graph_metadata`: Returns quantitative topology metrics (nodes, edges, density, latency).
    - Open `http://127.0.0.1:8001/kg_visualization`: Explore the interactive 3D physics Knowledge Graph!

---

## 4. Expected Review Panel Questions & Model Answers

**Q1: How do you handle unformatted student submissions across different file formats?**  
*Answer:* AIVA implements format-specialized parsers in `core/parser.py`. We stream text out of PDFs using `pdfplumber`, traverse document runs in `.docx` via `python-docx`, and extract slide presentation hierarchies in `.pptx` via `python-pptx`, unifying all extracted text into an immutable string stream before chunking.

**Q2: Why use Reciprocal Rank Fusion rather than adding dense cosine similarities and BM25 scores?**  
*Answer:* Dense cosine similarities and BM25 scores operate on fundamentally non-commensurate scales—cosine similarity is bounded in $[-1, 1]$ (or $[0, 1]$), while BM25 scores are unbounded term frequencies (e.g., $12.4$, $28.1$). Simple weighted addition requires query-time min-max normalization which is unstable on small candidate pools. RRF is scale-invariant because it operates purely on ordinal rank positions $r_m(d)$ with smoothing constant $k=60$.

**Q3: How do you guarantee the Knowledge Graph entities and relations are strictly typed?**  
*Answer:* We utilize Gemini's function calling structured output schema with Pydantic (`with_structured_output(GraphExtraction)`). This forces the model to emit deterministic JSON conforming strictly to `Node(id, type)` and `Relationship(source, target, type)`, eliminating unstructured hallucination or regex parsing failures.

**Q4: How does the Knowledge Graph improve the upcoming Viva evaluation phase (60% milestone)?**  
*Answer:* While flat RAG retrieves relevant document snippets, the Knowledge Graph exposes the relational topology of the student's project (e.g., which database connects to which service). The upcoming viva question generation agent uses graph traversals to formulate multi-hop diagnostic viva questions (e.g., "If ChromaDB locks up, how does Redis cache mitigation respond?").
