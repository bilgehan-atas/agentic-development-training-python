"""
EDGE ROUTING FUNCTIONS (used by conditional edges in graph.py).

A routing function looks at the state and returns the NAME of the next
node. It must not change the state - it only decides where to go.
"""

from .state import State


# ✅ GIVEN - example conditional edge: stop early if the server is down.
def route_after_load(state: State) -> str:
    return "report_error" if state.get("error") else "classify_intent"


# 🔲 TODO 1 - the routing function for the conditional edge after `classify_intent`.
#
# `classify_intent` stores one of "info" | "cancel" | "travel" | "unclear" in
# `state["intent"]`. Return the node that should handle it:
#
#   info    -> "answer_info"
#   cancel  -> "cancel_booking"
#   travel  -> "travel_agent"
#   unclear -> "ask_clarification"   (also use this when `intent` is missing)
#
# 👀 Look at `route_after_load` above for the pattern.
# ✅ Check: pytest tests/unit/test_1_conditional_edge.py
def route_by_intent(state: State) -> str:
    raise NotImplementedError("TODO 1: implement route_by_intent in src/flight_assistant/edges.py")
