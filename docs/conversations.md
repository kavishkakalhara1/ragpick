# Conversations

Real users don't ask one perfect question. They ask "How much does Lumen Plus cost?"
and then "and can I cancel it?". Retrieving for "and can I cancel it?" on its own finds
nothing useful. `chat()` fixes that:

```python
p = Pipeline.from_yaml("rag.yaml")
first = p.chat("How much does Lumen Plus cost?")
second = p.chat("and can I cancel it?", session_id=first.session_id)
print(second.standalone_question)  # "Can I cancel my Lumen Plus subscription?"
print(second.text, second.citations)
```

```bash
ragpick chat -c rag.yaml --groups support    # /reset for a new session, /exit to quit
```

## How it works

1. The session's recent turns are loaded.
2. The message is rewritten into a **standalone question**:
   - `rewrite: llm` (or `auto` with a real generator): the LLM rewrites it using the
     last few turns, keeping names, codes and numbers exactly.
   - `rewrite: heuristic` (or `auto` offline): follow-ups (short messages, messages
     starting with "and / what about / also …", or relying on "it / they / that") get
     the previous question attached as context.
   - `rewrite: off`: messages are used as-is.
3. The standalone question goes through the normal grounded pipeline: retrieval,
   access control, citations, tracing.
4. The turn is saved; only the last `max_turns` are kept.

```yaml
conversation:
  rewrite: auto
  max_turns: 10
  store: sqlite          # memory (default, LRU of max_sessions) | sqlite (survives restarts)
  ttl_seconds: 86400     # forget idle sessions after a day
```

The `conversational` blueprint is `hybrid` with `store: sqlite`. `chat()` works with
**every** blueprint and with routers (the router rewrites, then routes the standalone
question).

## Over HTTP

```bash
curl -s localhost:8000/chat -H "Authorization: Bearer $KEY" \
  -d '{"message": "How much does Lumen Plus cost?"}'
# → {"text": ..., "citations": [...], "session_id": "3f2a...", "standalone_question": ...}

curl -s localhost:8000/chat -H "Authorization: Bearer $KEY" \
  -d '{"message": "and can I cancel it?", "session_id": "3f2a..."}'

curl -s -X DELETE localhost:8000/sessions/3f2a... -H "Authorization: Bearer $KEY"
```

`/chat` accepts `groups`, `filters`, `top_k`, `tenant` and `"stream": true` (server-sent
events, with the final `answer` event carrying `session_id`).

**Sessions belong to their owner**, the API key plus tenant. Another key (or tenant)
gets **404 Unknown session** for someone else's session id, both when continuing and
when deleting, so history never leaks between clients. Session ids are 8–64 letters,
digits, `-` or `_`; new ones are random.
