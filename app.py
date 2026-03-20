import streamlit as st
import time
import os
from engine import run_assistant

# Page Config
st.set_page_config(page_title="AI Scientific Assistant Pro", page_icon="🔬", layout="wide")

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []

# Theme / Styling
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stTextInput > div > div > input { background-color: #1f2937; color: white; border-radius: 8px; }
    .stButton > button { background-color: #3b82f6; color: white; border-radius: 8px; transition: 0.3s; width: 100%; }
    .stButton > button:hover { background-color: #2563eb; transform: scale(1.02); }
    .result-card { padding: 20px; border-radius: 12px; background-color: #1f2937; border-left: 5px solid #3b82f6; margin-bottom: 20px;}
    .history-card { padding: 10px; border-radius: 8px; background-color: #2d3748; margin-bottom: 10px; font-size: 0.9em; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔬 AI Scientific Assistant Pro")
st.subheader("Agentic RAG with History & Multi-Step Reasoning")

# Sidebar: Knowledge & History
with st.sidebar:
    st.header("📂 Knowledge Base")
    uploaded_file = st.file_uploader("Teach the AI (Drop .txt files)", type=['txt'])
    if uploaded_file:
        with open(os.path.join("scientific_data", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Successfully added {uploaded_file.name} to AI brain!")
        st.info("The AI will automatically index this in the next query.")

    st.divider()
    st.header("🕰️ Research History")
    if not st.session_state.history:
        st.info("No previous analysis in this session.")
    else:
        for idx, turn in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Turn {len(st.session_state.history) - idx}"):
                st.markdown(f"**Q:** {turn['query']}")
                st.markdown(f"**A:** {turn['instrument']}")

    if st.button("Clear Session History"):
        st.session_state.history = []
        st.rerun()

# Main Interface
col_main, col_spacer = st.columns([3, 1])

with col_main:
    query = st.text_input("Describe your research problem", placeholder="e.g., 'I have a multi-layer gold film on a polymer substrate'")

    if st.button("Analyze Research Problem"):
        if query:
            with st.spinner("🧠 Reasoning (History-Aware)..."):
                try:
                    # Collect history for the engine
                    history_for_engine = [{"role": "user", "content": h['query']} for h in st.session_state.history]
                    
                    start_time = time.time()
                    result = run_assistant(query, history=history_for_engine)
                    end_time = time.time()
                    
                    # Update local history
                    st.session_state.history.append({
                        "query": query,
                        "instrument": result['instrument'],
                        "reasoning": result['reasoning'],
                        "preparation": result['preparation'],
                        "confidence": result['confidence'],
                        "retrieved": result['retrieved_docs']
                    })
                    
                    st.success(f"Dynamic Analysis complete in {end_time - start_time:.2f}s")
                    
                    # Display Results
                    st.markdown("### 📊 Latest Recommendation")
                    st.markdown(f"""
                    <div class="result-card">
                        <b>Primary Recommendation:</b> {result['instrument']}<br><br>
                        <b>Expert Reasoning:</b> {result['reasoning']}<br><br>
                        <b>Preparation Steps:</b> {result['preparation']}<br><br>
                        <b>Confidence Score:</b> <span style="color: #4ade80; font-weight: bold;">{result['confidence']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("🔍 Viewing Evidence (Retrieved Context)"):
                        for doc in result['retrieved_docs']:
                            st.write(f"- {doc}")
                            
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
        else:
            st.warning("Please enter a research problem to analyze.")

# Footer
st.divider()
st.markdown("Advanced Scientific Agentic RAG | Internship Portfolio Asset | 2024")
