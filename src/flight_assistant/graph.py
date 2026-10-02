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

    # ✅ TODO 1 (solved)
    builder.add_node("answer_info", answer_info)
    builder.add_node("cancel_booking", cancel_booking)
    builder.add_node("confirm_cancel", confirm_cancel)
    builder.add_node("travel_agent", travel_agent)
    builder.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        ["answer_info", "cancel_booking", "travel_agent", "ask_clarification"],
    )
    builder.add_edge("cancel_booking", "confirm_cancel")
    builder.add_edge("answer_info", END)
    builder.add_edge("confirm_cancel", END)
    builder.add_edge("travel_agent", END)

    # 🔲 TODO 4 - `interrupt()` needs a CHECKPOINTER: it saves the state after
    # every step so a paused run can be resumed later (same `thread_id`).
    # Pass `checkpointer=InMemorySaver()` to compile().
    return builder.compile()
