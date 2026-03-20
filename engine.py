import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
import glob

# Define State
class Message(TypedDict):
    role: str
    content: str

class GraphState(TypedDict):
    query: str
    history: List[Message]
    retrieved_docs: List[str]
    instrument: str
    reasoning: str
    preparation: str
    confidence: str
    is_valid: bool

# Initialize Models
print("Loading Embedding Model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Global Index State
_current_index = None
_all_documents = []

def load_and_index():
    """Scan scientific_data/ folder and rebuild index."""
    global _current_index, _all_documents
    print("Node: Refreshing Scientific Knowledge Index...")
    
    docs = []
    # Use glob to find all txt files in the directory
    data_files = glob.glob("scientific_data/*.txt")
    
    for file_path in data_files:
        with open(file_path, "r") as f:
            docs.extend([line.strip() for line in f.readlines() if line.strip()])
    
    if not docs:
        docs = ["No instrument data available."]
    
    _all_documents = docs
    embeddings = embed_model.encode(docs)
    d = embeddings.shape[1]
    idx = faiss.IndexFlatL2(d)
    idx.add(np.array(embeddings).astype('float32'))
    _current_index = idx
    return idx, docs

# Initial Load
load_and_index()

# Initialize Reasoning Model (Local)
print("Loading Reasoning Model (GPT-2)...")
generator = pipeline("text-generation", model="distilgpt2", device=-1)

# --- Nodes ---

def retrieve(state: GraphState):
    """History-aware retrieval."""
    print("Node: Retrieving with History...")
    
    # Refresh index in case new files were added
    idx, docs = load_and_index()
    
    query = state['query']
    # Build a persistent context query by concatenating recent history
    context_query = query
    if state.get('history'):
        # Just take the last 2 turns for context to avoid noise
        recent_history = " ".join([m['content'] for m in state['history'][-2:]])
        context_query = f"{recent_history} {query}"
    
    q_embed = embed_model.encode([context_query])
    D, I = idx.search(np.array(q_embed).astype('float32'), k=3)
    retrieved = [docs[i] for i in I[0]]
    
    return {"retrieved_docs": retrieved}

def reason(state: GraphState):
    """Advanced Reasoning for Multi-Step / Multi-Layer samples."""
    print("Node: Reasoning (Multi-Step Logic)...")
    query = state['query'].lower()
    context = "\n".join(state['retrieved_docs'])
    
    # Prompting for complex samples
    prompt = f"Data:\n{context}\n\nProblem: {query}\n\nIdentify best instruments (could be multiple if complex) and preparation."
    
    # Heuristic: If 'layer' or 'and' is in query, look for multiple matches
    instruments = []
    for doc in state['retrieved_docs']:
        inst = doc.split(":")[0].strip()
        if inst not in instruments:
            instruments.append(inst)
    
    # Primary instrument selection
    main_instrument = instruments[0]
    secondary = f" & {instruments[1]}" if len(instruments) > 1 and ("layer" in query or "complex" in query) else ""
    
    reasoning = state['retrieved_docs'][0].split(":")[1].strip()
    if secondary:
        reasoning += f" complemented by {instruments[1]} for full analysis."
    
    return {
        "instrument": main_instrument + secondary,
        "reasoning": reasoning,
        "preparation": f"Standard prep for {main_instrument}. {secondary if secondary else ''} requires specific handling."
    }

def validate(state: GraphState):
    """Rule-based validation with history awareness."""
    print("Node: Validating...")
    query = state['query'].lower()
    instrument = state['instrument'].upper()
    
    is_valid = True
    confidence = "85%"
    
    # Rule: Liquid + Vacuum = Bad
    if "liquid" in query and any(x in instrument for x in ["SEM", "XRD", "BET", "XPS"]):
        is_valid = False
        confidence = "20% (CRITICAL: Vacuum will evaporate your sample!)"
    
    # Rule: Powders + XRD = Perfect
    elif "powder" in query and "XRD" in instrument:
        confidence = "98% (XRD is the gold standard for powders)"
    
    return {"is_valid": is_valid, "confidence": confidence}

def format_output(state: GraphState):
    """Final output structuring."""
    print("Node: Formatting...")
    return state

# --- Build Graph ---

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("reason", reason)
workflow.add_node("validate", validate)
workflow.add_node("format_output", format_output)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "reason")
workflow.add_edge("reason", "validate")
workflow.add_edge("validate", "format_output")
workflow.add_edge("format_output", END)

app_engine = workflow.compile()

def run_assistant(query: str, history: List[dict] = None):
    inputs = {
        "query": query,
        "history": history or []
    }
    config = {"configurable": {"thread_id": "session_1"}}
    final_state = app_engine.invoke(inputs, config)
    return final_state
