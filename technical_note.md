# Technical Note: Advanced Agentic RAG & Dynamic Indexing

## Executive Summary
This project represents a sophisticated evolution of traditional RAG. By incorporating **Session History**, **Dynamic Vector Rebuilding**, and **Heuristic Multi-Step Reasoning**, this AI Scientific Assistant transitions from a simple search tool into an **Agentic Research Partner**.

## Advanced Architecture (v2.0)

### 1. Dynamic Knowledge Evolution
- **Auto-Indexing Engine**: The system now monitors the `scientific_data/` directory. When a new document is added (via the UI or manually), the RAG engine automatically re-triggers a FAISS index rebuild. This ensures the AI's "knowledge base" is always up-to-date without requiring code restarts.
- **Embedded Uploader**: The Streamlit interface directly hooks into this pipeline, allowing domain experts to "teach" the AI new scientific instruments simply by dropping text descriptions.

### 2. Session Awareness & Persistent Context
- **History-Aware Embedding Search**: Unlike standard RAG, which searches solely on the latest user query, this system maintains a rolling memory of the conversation. The embedding search query is dynamically constructed from both the current query and prior discourse, enabling follow-up questions like *"What about its crystal structure?"* without repeating the sample type.
- **Stateful Transition Logic**: Using LangGraph, the AI passes a `GraphState` object containing message history through its nodes (Retrieve -> Reason -> Validate), ensuring reasoning consistency across a multi-stage research dialogue.

### 3. Heuristic Multi-Step Reasoning
- **Complex Sample Logic**: The reasoning node is enhanced to detect keywords indicating multi-layered or multifaceted samples (e.g., "layered", "complex", "composite").
- **Complementary Instrument Synthesis**: Instead of returning a single result, the AI can now suggest a "primary" and "secondary" instrument combination, reasoning why both are necessary for a comprehensive analysis (e.g., SEM for imaging + XRD for structural analysis).

## Competitive Advantage for Internship
This level of implementation demonstrates:
- **Scalability**: Decoupling the data layer from the application logic.
- **Production Readiness**: Handling the real-world need for "hot-loading" data.
- **Advanced Orchestration**: Mastery of agentic patterns using LangGraph vs. standard linear scripts.
