"""
App: Streamlit UI for DBMS Concept Graph Tutor

Features:
- Chat-based DBMS Q&A
- RAG pipeline integration
- Concept graph visualization
- Multi-session chat history

Flow:
User Query → LangGraph Pipeline → Answer + Graph + Sources
"""

import streamlit as st
from graph_flow import compile_graph
from Nodes.safe_llm import safe_invoke
from pyvis.network import Network
import re, json

st.set_page_config(
    page_title="DBMS Concept Tutor",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:      #0d0f18;
    --surface: #13161f;
    --card:    #1a1e2e;
    --accent:  #00e5ff;
    --purple:  #7c4dff;
    --green:   #00e676;
    --yellow:  #ffab00;
    --red:     #ff5252;
    --text:    #e8eaf6;
    --muted:   #7986cb;
    --border:  #222540;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
#MainMenu, footer, header,
[data-testid="collapsedControl"],
[data-testid="stSidebar"] { display: none !important; }

/* Remove all default streamlit padding */
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stHorizontalBlock"] { gap: 0 !important; }

/* ── LEFT PANEL ── */
section.main > div.block-container > div > div > div > div:nth-child(1) {
    background: var(--surface) !important;
}
[data-testid="column"]:first-child {
    background: var(--surface);
    border-right: 1px solid var(--border);
    min-height: 100vh;
    padding: 1rem 0.8rem 2rem 0.8rem !important;
}
[data-testid="column"]:first-child > div {
    padding: 0 !important;
}

/* ── RIGHT PANEL ── */
[data-testid="column"]:last-child {
    padding: 1.2rem 2rem 2rem 2rem !important;
}
[data-testid="column"]:last-child > div {
    padding: 0 !important;
}

/* ── Topbar ── */
.topbar {
    display: flex;
    align-items: center;
    border-bottom: 2px solid var(--accent);
    padding-bottom: 0.5rem;
    margin-bottom: 1.2rem;
}
.topbar-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent);
    margin: 0;
    letter-spacing: -0.3px;
}
.topbar-title span { color: var(--purple); }

/* ── Panel title ── */
.panel-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.7rem;
}

/* ── Input ── */
.stTextInput > div > div > input {
    background: var(--card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,229,255,0.1) !important;
}

/* ── ALL buttons base ── */
.stButton > button {
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    transition: filter 0.15s !important;
}
.stButton > button:hover { filter: brightness(1.18) !important; }

/* Ask → button (last column) */
.ask-btn .stButton > button {
    background: linear-gradient(135deg, #7c4dff, #4527a0) !important;
    color: white !important;
    width: 100% !important;
    padding: 0.65rem 1rem !important;
}

/* New Chat button */
.new-chat-btn .stButton > button {
    background: linear-gradient(135deg, #00897b, #00695c) !important;
    color: white !important;
    width: 100% !important;
    padding: 0.42rem 0.8rem !important;
    font-size: 0.78rem !important;
    margin-bottom: 0.6rem !important;
}

/* Session list buttons */
.sess-btn .stButton > button {
    background: transparent !important;
    color: var(--muted) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 400 !important;
    text-align: left !important;
    padding: 0.38rem 0.65rem !important;
    width: 100% !important;
    margin-bottom: 0 !important;
}
.sess-btn-active .stButton > button {
    background: rgba(0,229,255,0.08) !important;
    color: var(--accent) !important;
    border-color: rgba(0,229,255,0.28) !important;
    font-weight: 600 !important;
}

/* ── Chat bubbles ── */
.bubble-q {
    background: rgba(124,77,255,0.1);
    border: 1px solid rgba(124,77,255,0.2);
    border-radius: 10px 10px 10px 2px;
    padding: 0.65rem 1rem;
    margin: 1rem 0 0.3rem 0;
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--text);
}
.bubble-a {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 2px 10px 10px 10px;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    line-height: 1.8;
    white-space: pre-wrap;
    color: var(--text);
}

/* ── Badges ── */
.tbadge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    padding: 0.13rem 0.48rem;
    border-radius: 4px;
    font-weight: 700;
    text-transform: uppercase;
    vertical-align: middle;
    margin-left: 0.4rem;
}
.t-definition   { background:rgba(0,230,118,.12); color:#00e676; border:1px solid rgba(0,230,118,.25);}
.t-comparison   { background:rgba(255,171,0,.12);  color:#ffab00; border:1px solid rgba(255,171,0,.25);}
.t-relationship { background:rgba(124,77,255,.12); color:#b388ff; border:1px solid rgba(124,77,255,.25);}
.t-process      { background:rgba(0,229,255,.12);  color:#00e5ff; border:1px solid rgba(0,229,255,.25);}
.t-sql          { background:rgba(255,82,82,.12);  color:#ff5252; border:1px solid rgba(255,82,82,.25);}

/* ── Source pills ── */
.pill {
    display: inline-block;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--muted);
    font-size: 0.67rem;
    font-family: 'Space Mono', monospace;
    padding: 0.13rem 0.48rem;
    border-radius: 4px;
    margin: 0.15rem 0.15rem 0 0;
}
.sec-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 0.7rem 0 0.3rem 0;
}
hr.div { border:none; border-top:1px solid var(--border); margin:1rem 0; }

/* ── Empty state ── */
.empty {
    text-align: center;
    padding: 4rem 1rem 2rem 1rem;
    color: var(--muted);
}
.empty .big  { font-family:'Space Mono',monospace; font-size:0.88rem; margin-bottom:0.7rem; }
.empty .small{ font-size:0.73rem; opacity:0.55; line-height:1.6; }

/* ── Sub-label under session button ── */
.sess-sub {
    font-size: 0.63rem;
    font-family: 'Space Mono', monospace;
    padding-left: 0.5rem;
    margin-bottom: 0.5rem;
    margin-top: 0.1rem;
    opacity: 0.65;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
GRAPH_TYPES = {"definition", "comparison", "relationship"}

# ── Helpers ───────────────────────────────────────────────────────────────────
def clean(answer):
    t = answer.replace("**Answer:**", "").strip()
    if "**Sources used:**" in t:
        t = t.split("**Sources used:**")[0].strip()
    return t

def tbadge(qtype):
    return f'<span class="tbadge t-{qtype}">{qtype}</span>'

def pills(sources):
    return "".join(f'<span class="pill">📎 {s}</span>' for s in sources)

def extract_edges(query, answer):
    prompt = f"""You are a DBMS knowledge graph extractor.
Extract 5-8 concept relationships from the answer.
Return ONLY a JSON array with keys "from", "label", "to".
- "from"/"to": short DBMS concept names (1-4 words)
- "label": short phrase like "is a type of", "depends on", "ensures", "uses"

Question: {query}
Answer: {answer[:700]}

Return only the JSON array, nothing else."""
    raw = safe_invoke(prompt)
    try:
        m = re.search(r'\[.*?\]', raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            return [(e["from"], e["label"], e["to"]) for e in data
                    if all(k in e for k in ("from","label","to"))]
    except Exception:
        pass
    return []

def build_graph(edges):
    net = Network(height="380px", width="100%", bgcolor="#1e2235",
                  font_color="#e8eaf6", directed=True)
    net.set_options("""{
      "nodes":{"shape":"dot","font":{"size":12,"face":"IBM Plex Sans","color":"#e8eaf6"},"borderWidth":2},
      "edges":{"font":{"size":9,"color":"#9fa8da","align":"middle"},
               "color":{"color":"#7c4dff","highlight":"#00e5ff"},
               "width":1.6,"arrows":{"to":{"enabled":true,"scaleFactor":0.6}},
               "smooth":{"type":"curvedCW","roundness":0.2}},
      "physics":{"barnesHut":{"gravitationalConstant":-8000,"springLength":130},"stabilization":{"iterations":100}}
    }""")
    palette = ["#00e5ff","#7c4dff","#00e676","#ffab00","#ff5252","#b388ff","#80cbc4"]
    seen = {}
    for src, lbl, tgt in edges:
        for name in (src, tgt):
            if name not in seen:
                i = len(seen); seen[name] = i
                net.add_node(name, label=name,
                             color=palette[i % len(palette)],
                             size=20 if i==0 else 14)
        net.add_edge(src, tgt, title=lbl, label=lbl)
    return net.generate_html()

# ── Session state ─────────────────────────────────────────────────────────────
# sessions: list of {id, name, messages:[{query,answer,qtype,sources,docs,edges}]}
# active_session: index of current session
if "pipeline" not in st.session_state:
    st.session_state.pipeline = compile_graph()
if "sessions" not in st.session_state:
    st.session_state.sessions = [{"id": 0, "name": "Session 1", "messages": []}]
if "active_session" not in st.session_state:
    st.session_state.active_session = 0
if "input_key" not in st.session_state:
    st.session_state.input_key = 0  # incrementing this clears the input

def get_session():
    return st.session_state.sessions[st.session_state.active_session]

def new_session():
    n = len(st.session_state.sessions) + 1
    st.session_state.sessions.append({"id": n-1, "name": f"Session {n}", "messages": []})
    st.session_state.active_session = len(st.session_state.sessions) - 1
    st.session_state.input_key += 1

# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 4], gap="small")

# ════════════════════════
#  LEFT — History panel
# ════════════════════════
with left:
    st.markdown('<div class="panel-title">💬 Chat History</div>', unsafe_allow_html=True)

    # New Chat button
    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("＋  New Chat", key="btn_new_chat"):
        new_session()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # List all sessions — newest first, only show, no sub-label clutter
    for i, sess in enumerate(reversed(st.session_state.sessions)):
        real_i = len(st.session_state.sessions) - 1 - i
        is_active = (real_i == st.session_state.active_session)

        msgs = sess["messages"]
        if msgs:
            # Use first question as the session name
            label = msgs[0]["query"][:34] + ("…" if len(msgs[0]["query"]) > 34 else "")
            count_txt = f" ({len(msgs)})"
        else:
            # New empty session — show "Untitled" until first question asked
            label = "Untitled"
            count_txt = ""

        css = "sess-btn-active" if is_active else "sess-btn"
        prefix = "▶ " if is_active else ""

        st.markdown(f'<div class="{css}">', unsafe_allow_html=True)
        if st.button(f"{prefix}{label}{count_txt}", key=f"sess_{real_i}"):
            st.session_state.active_session = real_i
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════
#  RIGHT — Main area
# ════════════════════════
with right:
    st.markdown('<div class="right-panel">', unsafe_allow_html=True)

    # Topbar with session name
    sess = get_session()
    sess_name = sess["name"]
    if sess["messages"]:
        sess_name = sess["messages"][0]["query"][:40]

    st.markdown(
        f'<div class="topbar">'
        f'<p class="topbar-title">🗄️ DBMS Concept <span>Graph Tutor</span></p>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Input row — key changes to clear input after submit
    in_col, btn_col = st.columns([5, 1])
    with in_col:
        query = st.text_input(
            "", placeholder="Ask a DBMS question…",
            label_visibility="collapsed",
            key=f"q_input_{st.session_state.input_key}"
        )
    with btn_col:
        st.markdown('<div class="ask-btn">', unsafe_allow_html=True)
        asked = st.button("Ask →", key="btn_ask")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Process question ──────────────────────────────────────────────────────
    if asked and query.strip():
        with st.spinner("Thinking…"):
            result = st.session_state.pipeline.invoke(
                {"query": query}, config={"recursion_limit": 50}
            )
        answer  = result.get("final_answer", "No answer.")
        qtype   = result.get("question_type", "definition")
        docs    = result.get("retrieved_docs", [])
        sources = sorted({d.metadata.get("source","unknown") for d in docs})
        text    = clean(answer)

        edges = []
        if qtype in GRAPH_TYPES:
            with st.spinner("Building concept graph…"):
                edges = extract_edges(query, text)

        # Append to CURRENT session
        sess_idx = st.session_state.active_session
        st.session_state.sessions[sess_idx]["messages"].append({
            "query": query, "answer": answer, "qtype": qtype,
            "sources": sources, "docs": docs, "edges": edges
        })
        # Auto-rename session to first question
        if len(st.session_state.sessions[sess_idx]["messages"]) == 1:
            st.session_state.sessions[sess_idx]["name"] = query[:34]

        # Clear input by bumping key
        st.session_state.input_key += 1
        st.rerun()

    # ── Display current session messages ─────────────────────────────────────
    sess = get_session()
    messages = sess["messages"]

    if not messages:
        st.markdown(
            '<div class="empty">'
            '<div class="big">⬆ Ask your first question for this session</div>'
            '<div class="small">'
            'Try: "What is BCNF?" · "Compare 2NF and 3NF" · "How does indexing work?"'
            '</div></div>',
            unsafe_allow_html=True
        )
    else:
        # Show ALL messages in this session (full conversation)
        for msg in messages:
            qtype   = msg["qtype"]
            sources = msg["sources"]
            edges   = msg["edges"]
            text    = clean(msg["answer"])

            # Question bubble
            st.markdown(
                f'<div class="bubble-q">'
                f'<span style="color:var(--muted);font-size:0.72rem;font-family:Space Mono,monospace;">Q</span> '
                f'{msg["query"]}{tbadge(qtype)}'
                f'</div>',
                unsafe_allow_html=True
            )

            # Answer bubble
            st.markdown(f'<div class="bubble-a">{text}</div>', unsafe_allow_html=True)

            # Sources + Graph
            src_col, graph_col = st.columns([1, 2])

            with src_col:
                st.markdown('<div class="sec-label"> Sources</div>', unsafe_allow_html=True)
                if sources:
                    st.markdown(pills(sources), unsafe_allow_html=True)
                    with st.expander("View snippets"):
                        for d in msg.get("docs", []):
                            snip = d.page_content[:250].replace("\n"," ")
                            src  = d.metadata.get("source","unknown")
                            st.markdown(f"**{src}**")
                            st.markdown(
                                f'<div style="font-size:0.78rem;color:#9fa8da;'
                                f'background:var(--bg);padding:0.4rem 0.6rem;'
                                f'border-radius:6px;margin-bottom:0.3rem">{snip}…</div>',
                                unsafe_allow_html=True
                            )
                else:
                    st.markdown('<span style="color:#7986cb;font-size:0.8rem">No sources.</span>',
                                unsafe_allow_html=True)

            with graph_col:
                if qtype in GRAPH_TYPES:
                    if edges:
                        st.markdown(
                            '<div style="font-family:Space Mono,monospace;font-size:0.8rem;'
                            'color:var(--purple);margin-bottom:0.2rem;"> Concept Graph</div>'
                            '<div style="font-size:0.7rem;color:var(--muted);margin-bottom:0.3rem;">'
                            'Nodes = concepts · Arrows = relationships</div>',
                            unsafe_allow_html=True
                        )
                        st.components.v1.html(build_graph(edges), height=400, scrolling=False)
                    else:
                        st.markdown(
                            '<div style="color:var(--muted);font-size:0.8rem;padding-top:0.5rem">'
                            'Could not extract concept relationships.</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown(
                        f'<div style="color:var(--muted);font-size:0.8rem;padding-top:0.5rem;line-height:1.8">'
                        f'No graph for <strong style="color:var(--red)">{qtype}</strong> questions.<br>'
                        f'Graph shows for: '
                        f'<span class="tbadge t-definition">definition</span> '
                        f'<span class="tbadge t-comparison">comparison</span> '
                        f'<span class="tbadge t-relationship">relationship</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            st.markdown('<hr class="div">', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
