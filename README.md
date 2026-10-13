# AIVA: 40% Milestone Submission (Review 1)
## Document Ingestion, Hybrid RAG & Knowledge Graph Engine

### 🌟 Implemented Features
1. **Document Ingestion (`core/parser.py`):** Multi-format parser for technical reports (PDF and DOCX).
2. **Hybrid RAG Retrieval (`core/rag_engine.py`):** ChromaDB dense vector store + Rank-BM25 sparse inverted index fused via Reciprocal Rank Fusion ($k=60$).
3. **Entity-Relation Knowledge Graph (`core/knowledge_graph.py`):** Automated entity extraction via Gemini, NetworkX graph assembly, and interactive 3D physics rendering via PyVis.
4. **FastAPI Microservice (`main.py`):** REST endpoints for report upload, hybrid queries, and graph visualization.
5. **Interactive Demonstration (`demo_40.py`):** Automated verification script.

---

### 🚀 Quick Start
```bash
cd AIVA40

# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure .env file with your GEMINI_API_KEY
# GEMINI_API_KEY="your-key"

# 3. Run the 40% verification demo
python demo_40.py

# 4. Or launch the 40% Review Backend server
python -m uvicorn main:app --port 8001 --reload
```
Interactive docs available at: `http://127.0.0.1:8001/docs`  
Interactive 3D Knowledge Graph available at: `http://127.0.0.1:8001/kg_visualization`
