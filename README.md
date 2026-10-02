# Super Memory — Neuro-Symbolic Cognitive Architecture

Super Memory is an enterprise-grade, hybrid **Neuro-Symbolic Cognitive Agent**. It unites modern Large Language Models (LLMs) with a formal **MeTTa Knowledge Graph (AtomSpace)** and an immutable **SQLite Evidence Ledger**, providing mathematically verifiable deduction, instant non-monotonic belief revision, and zero-hallucination factual grounding.

---

## 🏛️ System Architecture

```
User Input (Chat / Text Documents)
       │
       ▼
Next.js 16 (Turbopack Frontend) ───[REST API]───► FastAPI Backend (port 8000)
                                                        │
                                                        ▼
                                               NeuroSymbolicAgent
                                                        │
                      ┌─────────────────────────────────┴─────────────────────────────────┐
                      ▼                                                                   ▼
         TypeSafe Jev Decision API                                             LLM (OpenAI-Compatible)
    (Intent Routing & Semantic Guards)                                              (Tool Calling)
                      │                                                                   │
                      └─────────────────────────────────┬─────────────────────────────────┘
                                                        │
                                                        ▼
                                                MeTTa Engine
                                           (AtomSpace Hypergraph)
                                                        │
                                            ┌───────────┴───────────┐
                                            ▼                       ▼
                                   Backward Chaining        Belief Revision
                                    (Rule Deduction)       (Outdated State Retraction)
                                            │                       │
                                            └───────────┬───────────┘
                                                        │
                                                        ▼
                                            SQLite PersistenceManager
                                            (Immutable Evidence Ledger)
```

---

## 🔑 Required API Keys & Why JEV is Essential

To run Super Memory, create a `.env` file in the root directory (based on `.env.example`):

```bash
# 1. Primary Reasoning LLM (OpenAI-compatible)
OPENAI_API_KEY=your_openai_or_openrouter_or_asi_key
OPENAI_BASE_URL=https://api.openai.com/v1   # Or https://llm.c.singularitynet.io/v1 / OpenRouter
OPENAI_MODEL=gpt-4o-mini                    # Or openai/gpt-oss-20b

# 2. TypeSafe JEV Decision Engine (CRITICAL)
TYPESAFE_API_KEY=your_typesafe_or_openrouter_key
TYPESAFE_BASE_URL=https://openrouter.ai/api
```

### 🧠 Is the JEV Model Compulsory?
**YES, JEV (`typesafe/jev-1.13`) is an essential pillar of the architecture.**

While conventional chatbots send raw text directly into an LLM and hope for the best, Super Memory operates as a neuro-symbolic engine requiring formal logical constraints:
1. **Calibrated Intent Routing:** Before any LLM tokens are wasted, JEV deterministically classifies whether user input is a **Fact Assertion (`learn`)**, a **Knowledge Lookup (`query`)**, a **Belief Revision (`contradiction`)**, or an **Ambiguous Query (`clarification`)**.
2. **Predicate Alignment & Schema Critic:** When new facts are ingested, JEV enforces strict PascalCase relations and prevents predicate duplication or ontology explosion.
3. **Temporal Violation Guard:** Guarantees that raw dates and times never corrupt the primary relationship graph, mandating structured wrappers (`ValidDate`, `Recurrence`, `Exception`).
4. **Fallback Safety:** If `TYPESAFE_API_KEY` is temporarily missing, a deterministic regex fallback handles intent routing, but high-precision production alignment requires the JEV model.

---

## 🚀 Getting Started

### 1. Backend Setup (FastAPI & MeTTa)

```bash
# 1. Clone repository
git clone https://github.com/your-username/super-memory.git
cd super-memory

# 2. Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and TYPESAFE_API_KEY

# 5. Start the FastAPI backend
python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
The backend API is now live at `http://localhost:8000`.

---

### 2. Frontend Setup (Next.js Dashboard)

```bash
# In a new terminal, navigate to the frontend directory
cd frontend

# Install Node dependencies
npm install

# Start the Next.js Turbopack development server
npm run dev
```
The interface is now live at `http://localhost:3000`.

---

## 🛠️ Features & Capabilities

- **Live 2D/3D ForceGraph AtomSpace:** Real-time visual representation of all MeTTa nodes, entities, and predicates.
- **Explainable Trace Inspection:** Clicking any atom in the AtomSpace automatically highlights and scrolls to its original prompt in the chat widget.
- **Text Document Ingestion:** Attach any `.txt`, `.md`, `.csv`, `.log`, or `.json` document via the pin icon in the chatbot or the hero button. The engine parses the text, extracts factual statements, and grounds them into MeTTa atoms.
- **Auditable Evidence Ledger:** Immutable SQLite event ledger logging every addition (`ASSERT`) and retracted belief (`RETRACT`) with timestamps and context.
- **Backward-Chaining Deduction:** Proven logical deduction across multi-hop relations without manual rules.
- **Streamlit Alternative Interface:** The classic Streamlit GUI remains available via `streamlit run app.py`.

---

## 🔒 Security & Git Hygiene

- **Zero Hardcoded Secrets:** No API keys, passwords, or tokens are committed to this repository.
- `.env` and `.env.*` files are strictly excluded via `.gitignore`.
- Database files (`*.sqlite`, `*.db`) and transient logs are ignored. Always use environment variables for key management.
