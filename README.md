# Super Memory - Neuro-Symbolic Self-Evolving Agent

A hybrid Neuro-Symbolic AI agent that combines Large Language Models with a formal MeTTa Knowledge Graph (AtomSpace) for grounded, auditable, and self-evolving knowledge management.

## Architecture

```
User Input --> Streamlit UI --> NeuroSymbolicAgent
                                    |
                    +---------------+---------------+
                    |               |               |
              Jev Decision    LLM (OpenAI)    MeTTa Engine
              (Intent Route)  (Tool Calling)  (AtomSpace)
                    |               |               |
                    +-------+-------+       Backward Chainer
                            |               (Rule Inference)
                      PersistenceManager
                      (SQLite Evidence Ledger)
```

### Core Components

| Module | Purpose |
|--------|---------|
| `core/agent.py` | ReAct-powered conversational agent with tool calling |
| `core/metta_engine.py` | MeTTa AtomSpace with belief revision, backward chaining, temporal reasoning |
| `core/persistence.py` | SQLite evidence ledger with historical state tracking |
| `core/catalogue.py` | Entity catalogue with fuzzy search and disambiguation |
| `core/jev_adapter.py` | TypeSafe Jev Decision API for intent routing and predicate alignment |
| `core/clarification.py` | Uncertainty-aware dialogue management for ambiguous inputs |
| `core/models.py` | Strict dataclass definitions for the agent flow |
| `app.py` | Streamlit web interface |

### Key Capabilities

1. **Neuro-Symbolic Reasoning**: LLM translates natural language to formal MeTTa S-expressions; symbolic engine performs exact unification and backward chaining
2. **Belief Revision**: Single-valued predicates automatically retract outdated state when updated
3. **Backward Chaining**: Full AST-level rule inference with shared variable bindings across conjuncts
4. **Temporal Reasoning**: `ValidDate`, `Recurrence`, and `Exception` wrappers for date-scoped facts with exception overrides
5. **Historical Succession**: Permanent SQLite evidence ledger preserves all additions and removals, enabling predecessor/historical queries via `query_history`
6. **ReAct Cognitive Loop**: Agent follows Reason -> Act -> Observe workflow with structured tool use
7. **Jev Intent Routing**: TypeSafe AI decision engine for calibrated intent classification and entity disambiguation

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables in .env
# OPENAI_API_KEY=your_key
# OPENAI_BASE_URL=your_endpoint
# OPENAI_MODEL=your_model
# TYPESAFE_API_KEY=your_jev_key (optional)

# Run the app
streamlit run app.py

# Run acceptance tests
python tests/test_acceptance.py
```

## Project Structure

```
super memory/
  app.py                    # Streamlit web interface
  requirements.txt          # Python dependencies
  .env                      # API keys (not committed)
  core/
    __init__.py
    agent.py                # NeuroSymbolicAgent with ReAct loop
    metta_engine.py         # MeTTa engine + backward chainer
    persistence.py          # SQLite evidence ledger
    catalogue.py            # Entity catalogue + fuzzy search
    jev_adapter.py          # TypeSafe Jev Decision API adapter
    clarification.py        # Disambiguation ticket manager
    models.py               # Dataclass definitions
  data/
    evidence_ledger.sqlite  # Permanent claim store
    knowledge_base.metta    # Current AtomSpace snapshot
    audit_log.jsonl         # Append-only audit trail
  tests/
    test_acceptance.py      # 19 deterministic acceptance tests
```
