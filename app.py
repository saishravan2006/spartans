"""
app.py
🧠 Super Memory AI - Architecture V3 (Enterprise Neuro-Symbolic Cognitive Engine)
"""
from __future__ import annotations

import logging
import json
import time
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import streamlit as st

load_dotenv()

from core.metta_engine import MeTTaEngine
from core.persistence import PersistenceManager
from core.catalogue import EntityCatalogue
from core.agent import NeuroSymbolicAgent, SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page Configuration & Global Styling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    layout="wide",
    page_title="Super Memory AI — Neuro-Symbolic Engine",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# Enterprise Glassmorphic Theme Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 15% 10%, rgba(99, 102, 241, 0.08) 0%, transparent 40%),
                    radial-gradient(circle at 85% 85%, rgba(139, 92, 246, 0.08) 0%, transparent 45%),
                    #090d16;
        color: #e2e8f0;
    }

    /* Header Banner Styling */
    .hero-container {
        padding: 1.25rem 1.5rem;
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 24px -1px rgba(0, 0, 0, 0.3);
    }

    .hero-title {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.25rem;
    }

    .hero-badge {
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        background: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.4);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .hero-sub {
        font-size: 0.85rem;
        color: #94a3b8;
        margin: 0;
    }

    /* Metric Badges */
    .metric-pill-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 0.85rem;
    }

    .metric-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.75rem;
        border-radius: 10px;
        font-size: 0.78rem;
        font-weight: 500;
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.06);
        color: #cbd5e1;
    }

    .metric-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }

    .dot-green { background: #10b981; box-shadow: 0 0 8px #10b981; }
    .dot-purple { background: #8b5cf6; box-shadow: 0 0 8px #8b5cf6; }
    .dot-blue { background: #38bdf8; box-shadow: 0 0 8px #38bdf8; }
    .dot-amber { background: #f59e0b; box-shadow: 0 0 8px #f59e0b; }

    /* Glass Cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.55);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.85rem;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }

    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
    }

    .atom-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        background: rgba(30, 41, 59, 0.85);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 6px;
        padding: 0.4rem 0.6rem;
        color: #e0e7ff;
        margin-bottom: 0.4rem;
        word-break: break-all;
    }

    .rule-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        background: rgba(88, 28, 135, 0.25);
        border: 1px solid rgba(168, 85, 247, 0.35);
        border-radius: 6px;
        padding: 0.4rem 0.6rem;
        color: #f3e8ff;
        margin-bottom: 0.4rem;
        word-break: break-all;
    }

    /* Operation Badges */
    .op-badge-add {
        font-size: 0.68rem;
        font-weight: 700;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        text-transform: uppercase;
    }

    .op-badge-retract {
        font-size: 0.68rem;
        font-weight: 700;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        text-transform: uppercase;
    }

    /* Proof Container */
    .proof-box {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(147, 51, 234, 0.35);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
        font-size: 0.82rem;
    }

    /* Sidebar Clean styling */
    section[data-testid="stSidebar"] {
        background: rgba(10, 15, 28, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.4);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 6px 14px;
        font-weight: 500;
        font-size: 0.84rem;
        color: #94a3b8;
        border: none;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.25) !important;
        color: #ffffff !important;
        font-weight: 600;
        border: 1px solid rgba(99, 102, 241, 0.4) !important;
    }

    /* Chat bubble polish */
    div[data-testid="stChatMessage"] {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 14px !important;
        padding: 1rem !important;
        margin-bottom: 0.75rem !important;
        box-shadow: 0 2px 12px -2px rgba(0, 0, 0, 0.25) !important;
    }

    /* Expander styling */
    div[data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Session State & Engine Initialization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if "initialized" not in st.session_state:
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    engine = MeTTaEngine()
    pm = PersistenceManager(data_dir)
    pm.load_state(engine)
    
    catalogue = EntityCatalogue()
    catalogue.sync(engine.get_all_atoms())
    
    agent = NeuroSymbolicAgent(engine, catalogue, pm)
    
    st.session_state.engine = engine
    st.session_state.pm = pm
    st.session_state.catalogue = catalogue
    st.session_state.agent = agent
    
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    st.session_state.pending_prompt = None
    st.session_state.initialized = True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sidebar: Controls & Quick Benchmark Gauntlet
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("### ⚙️ System Controls")
    
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("🧹 Clear Chat", use_container_width=True, help="Clear active conversation history"):
            st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            st.session_state.pending_prompt = None
            st.rerun()
            
    with c_btn2:
        if st.button("🔄 Sync Graph", use_container_width=True, help="Re-sync catalogue with AtomSpace"):
            st.session_state.catalogue.sync(st.session_state.engine.get_all_atoms())
            st.success("Catalogue synced!")
            time.sleep(0.5)
            st.rerun()

    st.markdown("---")
    st.markdown("### 🎯 Quick Benchmark Gauntlet")
    st.caption("Click any one-shot scenario to test core neuro-symbolic guarantees:")

    # Gauntlet buttons that populate chat prompt
    if st.button("1️⃣ Belief Revision (Bob's Job)", use_container_width=True):
        st.session_state.pending_prompt = "Bob used to work at Google, but he works at OpenAI now. Where does Bob work?"
        st.rerun()

    if st.button("2️⃣ Symmetric Relation (Marriage)", use_container_width=True):
        st.session_state.pending_prompt = "Alice is married to Charlie. Who is Charlie married to?"
        st.rerun()

    if st.button("3️⃣ Temporal Exception (Lab 3 vs 5)", use_container_width=True):
        st.session_state.pending_prompt = "RoboticsClub meets in Lab3 on Fridays, but on 2026-10-09 they meet in Lab5. Where does RoboticsClub meet on 2026-10-09?"
        st.rerun()

    if st.button("4️⃣ Historical Succession (CEO)", use_container_width=True):
        st.session_state.pending_prompt = "John was CEO of Acme in 2021, and Sarah became CEO of Acme in 2023. Who was CEO of Acme in the past?"
        st.rerun()

    if st.button("5️⃣ Strict Layman Query", use_container_width=True):
        st.session_state.pending_prompt = "Where does Bob work currently, and can you explain how you know?"
        st.rerun()

    st.markdown("---")
    st.markdown("### 🛡️ Architecture Highlights")
    st.markdown("""
    - **MeTTa Hyperon Core**: Backward chaining & unification.
    - **Zero Hallucination**: Every claim grounded in AtomSpace.
    - **Layman Directive**: Translates all proofs to natural English.
    - **Audited Ledger**: SQLite transactional persistence.
    """)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Header Banner with Live Telemetry Pills
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
atoms = st.session_state.engine.get_all_atoms()
rules = getattr(st.session_state.engine, "_rules", [])
audit_log = st.session_state.pm.get_audit_log(100)
candidates = st.session_state.catalogue.candidates

st.markdown(f"""
<div class="hero-container">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
        <div>
            <div class="hero-title">
                <span>🧠 SUPER MEMORY AI</span>
                <span class="hero-badge">Architecture V3</span>
            </div>
            <p class="hero-sub">Enterprise Neuro-Symbolic Agent powered by OpenCog Hyperon MeTTa & Jev Intent Routing</p>
        </div>
    </div>
    <div class="metric-pill-container">
        <div class="metric-pill">
            <span class="metric-dot dot-green"></span>
            <span>Symbolic Engine: <strong>Active (Deterministic)</strong></span>
        </div>
        <div class="metric-pill">
            <span class="metric-dot dot-purple"></span>
            <span>AtomSpace: <strong>{len(atoms)} Atoms</strong></span>
        </div>
        <div class="metric-pill">
            <span class="metric-dot dot-blue"></span>
            <span>Evidence Ledger: <strong>{len(audit_log)} Transactions</strong></span>
        </div>
        <div class="metric-pill">
            <span class="metric-dot dot-amber"></span>
            <span>Catalogue: <strong>{len(candidates)} Entities / Predicates</strong></span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Layout: Conversational Stream (Left) & Symbolic Inspector (Right)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
col_chat, col_inspect = st.columns([1.5, 1.1], gap="medium")

# ─────────────────────────────────────────────────────────────────
# LEFT COLUMN: Conversational Experience & Proof Traces
# ─────────────────────────────────────────────────────────────────
with col_chat:
    st.markdown("#### 💬 Conversational Interface")
    
    # Message rendering container
    chat_container = st.container()
    with chat_container:
        for idx, msg in enumerate(st.session_state.messages):
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content")
                
                if role in ("user", "assistant") and content:
                    avatar = "👤" if role == "user" else "🧠"
                    with st.chat_message(role, avatar=avatar):
                        st.markdown(content)
                        
                elif role == "tool" and content:
                    try:
                        parsed = json.loads(content)
                        if "evidence" in parsed:
                            for ev in parsed["evidence"]:
                                if ev.get("origin") == "derived":
                                    with st.expander(f"✨ Verified Symbolic Proof (MeTTa AST)", expanded=False):
                                        st.markdown(f"**Target Binding:** `{ev.get('answer_bindings', 'None')}`")
                                        st.markdown(f"**Applied Rule:** `{ev.get('applied_rule', 'None')}`")
                                        st.markdown("**Premises Verified:**")
                                        for p in ev.get("premises", []):
                                            st.code(p, language="lisp")
                    except Exception:
                        pass
                        
            else:
                # Handle potential OpenAI SDK model object
                if getattr(msg, "role", "") == "assistant" and getattr(msg, "content", ""):
                    with st.chat_message("assistant", avatar="🧠"):
                        st.markdown(msg.content)

    # Handle pending prompt from Gauntlet button or Chat Input
    active_prompt = None
    if st.session_state.pending_prompt:
        active_prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
    else:
        active_prompt = st.chat_input("Speak naturally to ENI (e.g., 'Alice works at Acme Corp')...")

    if active_prompt:
        with st.chat_message("user", avatar="👤"):
            st.markdown(active_prompt)
            
        with st.spinner("🧠 Grounding in AtomSpace & formulating proof..."):
            reply = st.session_state.agent.chat(st.session_state.messages, active_prompt)
            
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(reply)
            
        st.rerun()


# ─────────────────────────────────────────────────────────────────
# RIGHT COLUMN: Deep Symbolic Inspector (Tabs)
# ─────────────────────────────────────────────────────────────────
with col_inspect:
    st.markdown("#### 🔬 Deep Symbolic Inspector")
    
    tab_atomspace, tab_ledger, tab_catalogue, tab_guarantees = st.tabs([
        f"🔮 AtomSpace ({len(atoms)})",
        f"📜 Ledger ({len(audit_log)})",
        f"🏷️ Catalogue ({len(candidates)})",
        "🛡️ Guarantees"
    ])

    # -------------------------------------------------------------
    # TAB 1: Live AtomSpace
    # -------------------------------------------------------------
    with tab_atomspace:
        st.caption("Live symbolic hypergraph maintained in OpenCog Hyperon AtomSpace:")
        
        search_atom = st.text_input("🔍 Filter Atoms / Rules...", placeholder="Search predicates or arguments...", label_visibility="collapsed")
        
        filtered_atoms = [a for a in atoms if search_atom.lower() in a.lower()] if search_atom else atoms
        filtered_rules = [r for r in rules if search_atom.lower() in r.lower()] if search_atom else rules

        if not filtered_atoms and not filtered_rules:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 2rem 1rem;">
                <p style="color: #94a3b8; margin-bottom: 0.5rem;">The AtomSpace is currently pristine and empty.</p>
                <p style="font-size: 0.8rem; color: #64748b;">Teach ENI facts in the chat or click a Gauntlet test button on the sidebar.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            if filtered_rules:
                st.markdown(f"**Inference Rules ({len(filtered_rules)})**")
                for r in filtered_rules:
                    st.markdown(f'<div class="rule-pill">⚙️ {r}</div>', unsafe_allow_html=True)
                    
            if filtered_atoms:
                st.markdown(f"**Grounded Facts ({len(filtered_atoms)})**")
                for a in filtered_atoms:
                    st.markdown(f'<div class="atom-pill">🔹 {a}</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # TAB 2: SQLite Evidence Ledger
    # -------------------------------------------------------------
    with tab_ledger:
        st.caption("Immutable chronological transaction ledger with idempotent hash verification:")
        
        search_ledger = st.text_input("🔍 Filter Ledger...", placeholder="Filter by user utterance or fact...", label_visibility="collapsed")
        
        filtered_log = audit_log
        if search_ledger:
            filtered_log = [
                entry for entry in audit_log
                if search_ledger.lower() in entry.get("user_input", "").lower()
                or search_ledger.lower() in entry.get("diff", "").lower()
            ]

        if not filtered_log:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 2rem 1rem;">
                <p style="color: #94a3b8;">No audited claims found in SQLite ledger.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for entry in reversed(filtered_log[-20:]):
                op = entry.get("operation", "add").lower()
                badge_class = "op-badge-add" if op == "add" else "op-badge-retract"
                badge_text = "ASSERTION" if op == "add" else "RETRACTION"
                
                st.markdown(f"""
                <div class="glass-card" style="padding: 0.75rem 0.85rem; margin-bottom: 0.5rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                        <span class="{badge_class}">{badge_text}</span>
                        <span style="font-size: 0.72rem; color: #64748b;">{entry.get('timestamp', '')[:19].replace('T', ' ')} UTC</span>
                    </div>
                    <div style="font-size: 0.82rem; color: #cbd5e1; margin-bottom: 0.35rem;">
                        <em>"{entry.get('user_input', '')}"</em>
                    </div>
                    <div class="atom-pill" style="margin-bottom: 0; font-size: 0.76rem;">
                        {entry.get('diff', '')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # TAB 3: Entity & Schema Catalogue
    # -------------------------------------------------------------
    with tab_catalogue:
        st.caption("Harmonized entities and schema predicates extracted via universal AST walking:")
        
        filter_type = st.radio("Catalogue View", ["All", "Predicates", "Entities"], horizontal=True, label_visibility="collapsed")
        
        items = list(candidates.values())
        if filter_type == "Predicates":
            items = [c for c in items if c.is_predicate]
        elif filter_type == "Entities":
            items = [c for c in items if not c.is_predicate]

        if not items:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 2rem 1rem;">
                <p style="color: #94a3b8;">Catalogue is currently empty.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for c in items:
                badge_color = "rgba(139, 92, 246, 0.2)" if c.is_predicate else "rgba(6, 182, 212, 0.2)"
                border_color = "rgba(139, 92, 246, 0.4)" if c.is_predicate else "rgba(6, 182, 212, 0.4)"
                text_color = "#c084fc" if c.is_predicate else "#67e8f9"
                tag_name = "PREDICATE" if c.is_predicate else "ENTITY"
                
                alias_str = ", ".join(list(c.aliases)[:3]) if c.aliases else "none"
                
                st.markdown(f"""
                <div class="glass-card" style="padding: 0.65rem 0.85rem; margin-bottom: 0.4rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 600; font-size: 0.84rem; color: #f1f5f9;">{c.canonical_id}</span>
                        <span style="font-size: 0.65rem; font-weight: 700; padding: 0.15rem 0.4rem; border-radius: 4px; background: {badge_color}; color: {text_color}; border: 1px solid {border_color};">
                            {tag_name}
                        </span>
                    </div>
                    <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.2rem;">
                        Label: <em>{c.display_label}</em> &bull; Aliases: <em>{alias_str}</em>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # TAB 4: Core Guarantees Overview
    # -------------------------------------------------------------
    with tab_guarantees:
        st.caption("The 6 Cognitive Guarantees enforced by Super Memory V3:")
        
        guarantees = [
            ("⚡ 1. Zero Hallucination", "Every generated answer is strictly unified against grounded MeTTa hypergraph atoms. If unsupported, the engine reports NO_SUPPORTING_CLAIM."),
            ("🔄 2. Non-Monotonic Belief Revision", "Outdated or superseded facts are automatically retracted from the active AtomSpace while preserving audit history in SQLite."),
            ("⏱️ 3. Temporal Scoping & Exceptions", "Distinguishes between recurring defaults and localized date exceptions using first-class (ValidDate ...), (Recurrence ...), and (Exception ...) AST forms."),
            ("🏛️ 4. Historical Succession Ledger", "Past states are permanently queryable via SQLite history, enabling deep queries like 'Who was CEO before the current one?'"),
            ("🔗 5. Symmetric & Multi-Hop Deduction", "Bi-directional and transitive relationships are proven at query time via strict MeTTa backward-chaining rules without data duplication."),
            ("🗣️ 6. Layman 'Hood Down' Directive", "Internal S-expressions, AST bindings, and database details are concealed behind crisp, natural, conversational prose.")
        ]
        
        for title, desc in guarantees:
            st.markdown(f"""
            <div class="glass-card" style="padding: 0.75rem 0.9rem; margin-bottom: 0.5rem;">
                <div style="font-weight: 600; font-size: 0.84rem; color: #38bdf8; margin-bottom: 0.25rem;">{title}</div>
                <div style="font-size: 0.77rem; color: #94a3b8; line-height: 1.4;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
