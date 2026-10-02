"""
Human-in-the-loop approval for actions that move money.

Called by the `confirm_cancel` node right before it hits the flight server.
(The travel agent's tools are guarded by `HumanInTheLoopMiddleware` instead,
see `interrupt_on` in tools.py - same `interrupt()` mechanism under the hood.)
"""

from langgraph.types import interrupt

# 🔲 TODO 4 - right now every action is auto-approved. Make it ask a human:
#   1. answer = interrupt({"question": f"{action} Approve? (y/n)"})
#      (main.py already prints `question` and resumes with what you type.)
#   2. Return True only for "y" / "yes" (case-insensitive).
#   3. In graph.py, compile the graph with a checkpointer (see the TODO 4 note there).
# ✅ Check: pytest tests/unit/test_4_human_approval.py
#    Try:   python -m flight_assistant.main "Cancel ABC123"  → answer "n" and check nothing was cancelled
#
# ⚠️ On resume, LangGraph re-runs the WHOLE node that called `interrupt()`, from
# its first line. So everything a node does before asking must be safe to repeat:
# no LLM calls (they may answer differently the second time), no API calls.
# That's why cancelling is split into `cancel_booking` (LLM) → `confirm_cancel` (ask + act).
def require_approval(action: str) -> bool:
    return True
