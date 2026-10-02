"""
EDGE ROUTING FUNCTIONS (used by conditional edges in graph.py).

A routing function looks at the state and returns the NAME of the next
node. It must not change the state - it only decides where to go.
"""

from .state import State


# ✅ GIVEN - example conditional edge: stop early if the server is down.
def route_after_load(state: State) -> str:
    return "report_error" if state.get("error") else "classify_intent"


# ✅ TODO 1 (solved)
def route_by_intent(state: State) -> str:
    return {
        "info": "answer_info",
        "cancel": "cancel_booking",
        "travel": "travel_agent",
    }.get(state.get("intent"), "ask_clarification")
