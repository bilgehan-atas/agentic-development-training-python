"""
Human-in-the-loop approval for actions that move money.

Called by the `confirm_cancel` node right before it hits the flight server.
(The travel agent's tools are guarded by `HumanInTheLoopMiddleware` instead,
see `interrupt_on` in tools.py - same `interrupt()` mechanism under the hood.)
"""

import re

from langgraph.types import interrupt

IS_YES = re.compile(r"^\s*y(es)?\s*$", re.IGNORECASE)


# ✅ TODO 4 (solved)
def require_approval(action: str) -> bool:
    answer = interrupt({"question": f"{action} Approve? (y/n)"})
    if answer is True:
        return True
    return bool(IS_YES.match(str(answer)))
