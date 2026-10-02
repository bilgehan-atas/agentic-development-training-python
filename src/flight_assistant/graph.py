"""
GRAPH ASSEMBLY: nodes + edges + conditional edges.

  .add_node(name, fn)                             a NODE (a step; see nodes.py)
  .add_edge(from_, to)                            an EDGE: after `from_`, ALWAYS go to `to`
  .add_conditional_edges(from_, router, [...])    a CONDITIONAL EDGE: after `from_`,
                                                   call `router(state)` to pick the next node

START and END are special built-in nodes: where a run begins and finishes.

Print the graph as a Mermaid diagram with `python -m flight_assistant.draw`.
"""

from langgraph.graph import END, START, StateGraph

from .edges import route_after_load, route_by_intent
from .nodes import (
    answer_info,
    ask_clarification,
    cancel_booking,
    classify_intent,
    confirm_cancel,
    load_context,
    report_error,
    travel_agent,
)
from .state import State


def build_graph():
    builder = StateGraph(State)
    builder.add_node("load_context", load_context)
    builder.add_node("report_error", report_error)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("ask_clarification", ask_clarification)

    builder.add_edge(START, "load_context")

    # ✅ GIVEN - example conditional edge
    builder.add_conditional_edges("load_context", route_after_load, ["classify_intent", "report_error"])

    builder.add_edge("report_error", END)
    builder.add_edge("ask_clarification", END)

    # 🔲 TODO 1 - make the graph branch on the user's intent.
    #   a) Register 4 more NODES with builder.add_node(name, fn):
    #        "answer_info" -> answer_info, "cancel_booking" -> cancel_booking,
    #        "confirm_cancel" -> confirm_cancel, "travel_agent" -> travel_agent
    #        (all imported above)
    #   b) Replace the plain edge below with a CONDITIONAL EDGE from
    #      "classify_intent" that calls `route_by_intent` (src/flight_assistant/edges.py) and can
    #      go to: "answer_info", "cancel_booking", "travel_agent", "ask_clarification".
    #      👀 Copy the `load_context` conditional edge above.
    #   c) Add plain EDGES with builder.add_edge(...):
    #        "cancel_booking" -> "confirm_cancel"   (a 2-step workflow, see TODO 3)
    #        "answer_info", "confirm_cancel", "travel_agent" -> END
    #
    # Right now every request ends up in ask_clarification. LangGraph refuses
    # to compile a graph with unreachable nodes - that's why these nodes are
    # not registered yet. Run `python -m flight_assistant.draw` before and after!
    # ✅ Check: pytest tests/unit/test_1_graph_wiring.py
    builder.add_edge("classify_intent", "ask_clarification")

    # 🔲 TODO 4 - `interrupt()` needs a CHECKPOINTER: it saves the state after
    # every step so a paused run can be resumed later (same `thread_id`).
    # Pass `checkpointer=InMemorySaver()` to compile().
    return builder.compile()
