"""All LLM-judge prompts live in this file.

Every judge answers with a single JSON object: ``{"score": <0..1>, "reason": "<short>"}``.
"""

from __future__ import annotations

JUDGE_SYSTEM = (
    "You are a strict, impartial evaluator of question-answering systems. "
    'Respond with a single JSON object and nothing else: {"score": <number between 0 and 1>, '
    '"reason": "<one short sentence>"}.'
)

FAITHFULNESS_PROMPT = """Evaluate FAITHFULNESS: is every claim in the answer supported by the \
context passages? Ignore whether the answer is helpful; only judge support.

Scoring:
- 1.0: every claim is directly supported by the context.
- 0.5: some claims are supported, others are not.
- 0.0: the main claims are unsupported or contradict the context.
An answer that says "I don't know" makes no claims and scores 1.0.

Context:
{contexts}

Answer:
{answer}
"""

ANSWER_RELEVANCE_PROMPT = """Evaluate ANSWER RELEVANCE: does the answer address the question \
directly and completely? Ignore whether it is factually correct.

Scoring:
- 1.0: fully and directly answers the question.
- 0.5: partially answers it or includes a lot of irrelevant content.
- 0.0: does not address the question (an "I don't know" answer scores 0.0).

Question:
{question}

Answer:
{answer}
"""
