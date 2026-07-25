# Modern LLM Architecture

The simulator now treats LLMs as high-level reasoning components rather than as a utility for every small transformation.

## Provider configuration

Configure providers in `reverie/backend_server/utils.py` (copy from `utils.py.example`). The default example targets Ollama's OpenAI-compatible endpoint:

```python
llm_provider = "ollama"
llm_api_base = "http://localhost:11434/v1"
llm_chat_model = "mistral:7b"
llm_embedding_model = "nomic-embed-text"
```

OpenAI-compatible providers can use `llm_provider = "openai"` or keep their own API base/key settings.

## Observability

Set `llm_trace_enabled = True` to write JSONL request traces to `llm_trace_path` (default `llm_trace.jsonl`). Records include provider, model, call kind, elapsed time, input length, and success/failure. Tracing is off by default and should not create files when disabled.

Prompt templates are cached in-process. Embeddings are cached by `(provider, embedding_model, normalized_text)`, with optional persistence via `llm_embedding_cache_path`. This prevents mixing OpenAI and Ollama embedding vectors. Retrieval still returns zero similarity for dimension mismatches so older bundled simulations with OpenAI embeddings do not crash.

## Feature flags

High-risk behavior remains configurable in `utils.py`:

```python
llm_generate_emojis = False
llm_generate_event_triples = False
llm_score_poignancy = False
llm_compile_actions = False
structured_daily_planning = True
batched_conversations = True
structured_reflection = True
```

Set fallback flags to `True` only when comparing against legacy LLM-heavy behavior.

## Cognitive pipeline responsibilities

Code should handle mechanics: pathing, object validation, schedule repair, emojis, event triples, and basic poignancy scoring.

Retrieval/memory should handle context selection, relationship summaries, and embeddings.

LLMs should be reserved for high-level planning, bounded dialogue, reflection, and ambiguous social judgment. Avoid reintroducing sequential micro-calls in hot paths.
