"""
AIVA 40% Milestone: Document Ingestion, Hybrid RAG, and Knowledge Graph Engine
Phase: Review 1 (40% Milestone)
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import json
from typing import List, Optional
from dotenv import load_dotenv

from core.parser import DocumentParser
from core.knowledge_graph import KGBuilder
from core.rag_engine import RAGEngine

load_dotenv()

app = FastAPI(
    title="AIVA - 40% Review Backend",
    description="Document Ingestion, Hybrid Dense/Sparse RAG, and Entity-Relation Knowledge Graph extraction."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engines for 40% phase
rag_engine = None
kg_builder = None
session_metadata = {
    "document_name": "",
    "nodes": [],
    "relationships": [],
    "chunks_indexed": 0
}

@app.on_event("startup")
def startup():
    global rag_engine, kg_builder
    rag_engine = RAGEngine()
    kg_builder = KGBuilder()
    print("AIVA 40% Engine: Hybrid RAG & Knowledge Graph Engine initialized.")

@app.get("/")
def root():
    return {
        "status": "online",
        "milestone": "40% Submission (Review 1)",
        "modules": [
            "Document Parser (PDF/DOCX)",
            "Hybrid RAG (Dense ChromaDB + Sparse BM25 + RRF k=60)",
            "Entity-Relation Knowledge Graph (Gemini + NetworkX + PyVis 3D)"
        ]
    }

@app.post("/upload_and_build_kg")
async def upload_and_build_kg(file: UploadFile = File(...)):
    """Uploads technical report, chunks text, builds Knowledge Graph, and indexes into Hybrid RAG."""
    tmp_path = f"temp_{file.filename}"
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        session_metadata["document_name"] = file.filename
        
        # 1. Parse document text
        parser = DocumentParser()
        text = parser.extract_text(tmp_path)
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from document.")
            
        # 2. Reset vector database & index hybrid RAG
        rag_engine.clear()
        chunks_indexed = rag_engine.index_document(text, document_id=file.filename)
        session_metadata["chunks_indexed"] = chunks_indexed
        
        # 3. Extract Knowledge Graph via Gemini
        extraction = kg_builder.extract_graph_from_text(text)
        if not extraction:
            raise HTTPException(status_code=500, detail="Failed to extract knowledge graph.")
            
        # 4. Construct NetworkX graph and render PyVis 3D HTML
        kg_builder.build_networkx_graph(extraction)
        output_html = kg_builder.generate_html_visualization("kg_visualization.html")
        
        session_metadata["nodes"] = [node.model_dump() for node in extraction.nodes]
        session_metadata["relationships"] = [rel.model_dump() for rel in extraction.relationships]
        
        # Save graph metadata for dissertation reporting
        graph_meta = {
            "document_name": file.filename,
            "node_count": len(session_metadata["nodes"]),
            "relationship_count": len(session_metadata["relationships"]),
            "chunks_indexed": chunks_indexed
        }
        with open("graph_metadata.json", "w", encoding="utf-8") as f:
            json.dump(graph_meta, f, indent=2)
            
        return {
            "message": "Report parsed, Knowledge Graph built, and Hybrid RAG indexed successfully!",
            "document_name": file.filename,
            "node_count": len(session_metadata["nodes"]),
            "relationship_count": len(session_metadata["relationships"]),
            "chunks_indexed": chunks_indexed,
            "visualization_path": output_html
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/query_rag")
def query_rag(query: str = Query(..., description="Query string to retrieve from report")):
    """Tests Hybrid RAG retrieval combining Dense Semantic search and Sparse BM25 via RRF."""
    if not rag_engine or session_metadata["chunks_indexed"] == 0:
        return {"error": "No document indexed yet. Please upload a report first."}
        
    results = rag_engine.retrieve(query, top_k=3)
    return {
        "query": query,
        "results_count": len(results),
        "fused_results": results
    }

@app.get("/kg_visualization", response_class=HTMLResponse)
def get_kg_visualization():
    """Renders the 3D physics-based interactive Knowledge Graph in the browser."""
    if os.path.exists("kg_visualization.html"):
        with open("kg_visualization.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h3>No Knowledge Graph found. Please upload a report first.</h3>", status_code=404)

@app.get("/graph_metadata")
def get_graph_metadata():
    """Returns dissertation graph metrics (nodes, edges, chunks)."""
    if os.path.exists("graph_metadata.json"):
        with open("graph_metadata.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return session_metadata
