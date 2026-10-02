"""
Runs the graph once for a user message and prints each step as it happens.
Handles human-in-the-loop interrupts by asking `approve()` and resuming.

Two kinds of interrupt can show up:
  - `{"question": ...}`        from `require_approval()` (approval.py) -> resumed with the raw answer
  - `{"action_requests": ...}` from the travel agent's `HumanInTheLoopMiddleware`
                                -> one question per tool call, resumed with `{"decisions": [...]}`
"""

import re
from typing import Any, Callable, Optional

from langgraph.types import Command

Approver = Callable[[str], str]

IS_YES = re.compile(r"^\s*y(es)?\s*$", re.IGNORECASE)


def _is_hitl_request(value: Any) -> bool:
    return isinstance(value, dict) and "action_requests" in value


def _answer_interrupt(value: Any, approve: Approver) -> Any:
    """Asks the human about one interrupt and returns the value to resume it with."""
    if not _is_hitl_request(value):
        return approve(value["question"])

    decisions = []
    for action in value["action_requests"]:
        description = action.get("description") or f"Run {action['name']}."
        answer = approve(f"{description} Approve? (y/n)")
        if IS_YES.match(answer):
            decisions.append({"type": "approve"})
        else:
            decisions.append({"type": "reject", "message": f"The user rejected {action['name']}; it was NOT done."})
    return {"decisions": decisions}


def _summarize(update: Optional[dict]) -> str:
    """Short description of a node's state update, e.g. `intent=travel`."""
    if not update:
        return ""
    parts = []
    for key, value in update.items():
        if key in ("messages", "user") or value is None:
            continue
        parts.append(f"{key}={value if isinstance(value, str) else value!r}")
    return " ".join(parts)


def run_turn(app, text: str, thread_id: str, approve: Approver, log: Callable[[str], None] = print) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    graph_input: Any = {"messages": [{"role": "user", "content": text}]}
    final_state: Optional[dict] = None

    while True:
        pending: list[tuple[str, Any]] = []
        for mode, chunk in app.stream(graph_input, config, stream_mode=["updates", "values"]):
            if mode == "values":
                final_state = chunk
                continue
            for node, update in chunk.items():
                if node == "__interrupt__":
                    pending.extend((i.id, i.value) for i in update)
                else:
                    log(f"  → {node} {_summarize(update)}".rstrip())
        if not pending:
            break

        # Several interrupts can be pending at once: answer each, keyed by its id.
        resume = {interrupt_id: _answer_interrupt(value, approve) for interrupt_id, value in pending}
        graph_input = Command(resume=resume)

    assert final_state is not None
    return final_state
