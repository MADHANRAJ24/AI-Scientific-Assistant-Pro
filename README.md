# AI Scientific Assistant Pro (Agentic RAG) 🔬

An advanced, stateful RAG-based system using LangGraph for multi-stage scientific reasoning.

## 🚀 Advanced Features (v2.0)
- **Persistent Session History**: The AI remembers your previous questions. Ask follow-up questions like *"What about its chemical bonds?"* without re-explaining the sample.
- **Dynamic Knowledge Injection**: Add new scientific instruments on the fly! Just drop a `.txt` file into the `scientific_data/` folder or use the **"Teach the AI"** section in the UI.
- **Multi-Step Reasoning**: Capable of recommending instrument combinations for complex multi-layered samples.
- **Rule-Based Validation Guardrails**: Automatically warns about physical constraints (e.g., vacuum compatibility issues for liquids).

## 🛠️ Tech Stack
- **LangGraph**: Stateful agentic workflow.
- **FAISS**: Dynamic vector database search.
- **Sentence Transformers**: `all-MiniLM-L6-v2` embeddings.
- **Transformers**: Local `distilgpt2` for reasoning.
- **Streamlit**: Advanced interactive dashboard.

## 📦 Navigation
- `engine.py`: Core Agentic RAG logic.
- `app.py`: Streamlit UI hub.
- `scientific_data/`: Dynamic knowledge source (drop files here!).
- `technical_note.md`: Detailed architecture review (for portfolio).

## 🚀 Quick Start
1. `pip install -r requirements.txt`
2. `streamlit run app.py`

## 📊 Example Advanced Query
1. "I have a solid gold thin film." -> Suggests SEM.
2. "What about its crystal structure?" -> Suggests XRD (Retains history of 'gold sample').
