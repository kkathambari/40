# Review 1 Presentation & Viva Defense Guide (40% Milestone)

---

## 1. Milestone Focus & Objectives
By the 40% milestone, the system must prove that it can take any unformatted student technical report (PDF/DOCX), parse its content, extract deep semantic relationships into a Knowledge Graph, and provide fast, grounded hybrid retrieval.

---

## 2. Core Technical Concepts to Explain to the Panel

### A. Why Hybrid RAG instead of Pure Vector Search?
*   **Vector Search limitation:** Dense vector embeddings (cosine distance on high-dimensional vectors) are great for fuzzy semantic matches, but often fail to match exact code variables, version numbers, or acronyms (e.g. "RRF k=60", "port 8001").
*   **Sparse BM25 limitation:** BM25 is purely lexical/probabilistic; it cannot match synonyms or conceptual paraphrasing.
*   **AIVA Solution:** Fuses both using **Reciprocal Rank Fusion (RRF)**:
    $$\text{RRF\_Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
    Where:
    - $M$ is the set of rankers (Dense ChromaDB and Sparse BM25).
    - $r_m(d)$ is the rank of document $d$ in system $m$.
    - $k = 60$ is the standard smoothing constant preventing low-ranked outliers from dominating.

### B. Knowledge Graph Extraction Architecture
*   Uses Google Gemini with structured Pydantic schemas (`Entity`, `Relationship`).
*   Entities are tagged by type: `Concept`, `Framework`, `Algorithm`, `Tool`, `Database`.
*   Relationships define architecture: `depends_on`, `implements`, `fuses`, `evaluates`.
*   Visualized as a physics-based 3D force-directed graph via `NetworkX` and `PyVis`.

---

## 3. How to Demonstrate in Front of Reviewers
1.  **Run the demo script:**
    ```bash
    python demo_40.py
    ```
2.  **Or launch the FastAPI server:**
    ```bash
    python -m uvicorn main:app --port 8001 --reload
    ```
3.  Open browser to `http://127.0.0.1:8001/docs`:
    - Execute `/upload_and_build_kg` with any sample PDF.
    - Execute `/query_rag?query=what algorithms are used` to show the fused RRF ranking.
    - Open `http://127.0.0.1:8001/kg_visualization` to show the interactive 3D physics graph!

---

## 4. Expected Review Panel Questions & Model Answers

**Q1: How do you prevent out-of-memory errors when processing large 100-page dissertations?**  
*Answer:* We use streaming page extraction via `pdfplumber` combined with LangChain's recursive character text splitter. Documents are partitioned into fixed 500-token chunks with a 50-token contextual sliding window overlap before batch-inserting into ChromaDB.

**Q2: Why use Reciprocal Rank Fusion rather than simple score addition?**  
*Answer:* Dense cosine similarities and BM25 scores operate on entirely different mathematical scales (cosine similarity is in $[-1, 1]$ or $[0, 1]$, while BM25 scores can be unbounded, e.g. $14.8$ or $22.3$). Normalizing them requires calculating min-max bounds on every query. RRF bypasses score-scale incompatibility entirely by relying purely on ordinal rank positions.
