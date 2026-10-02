from typing import Optional, TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from flight_assistant.approval import require_approval
from flight_assistant.graph import build_graph


class ApprovalState(TypedDict, total=False):
    approved: Optional[bool]


def approval_graph():
    """A tiny graph with a single node that asks for approval."""
    builder = StateGraph(ApprovalState)
    builder.add_node("ask", lambda state: {"approved": require_approval("Cancel booking ABC123.")})
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    return builder.compile(checkpointer=InMemorySaver())


class TestHumanApproval:
    """TODO 4 — human-in-the-loop approval"""

    def test_require_approval_pauses_graph_with_interrupt(self):
        graph = approval_graph()
        config = {"configurable": {"thread_id": "t1"}}

        paused = graph.invoke({}, config)

        assert "__interrupt__" in paused
        interrupts = paused["__interrupt__"]
        assert len(interrupts) == 1
        assert "ABC123" in str(interrupts[0].value)

    @pytest.mark.parametrize(
        "answer,approved",
        [("y", True), ("yes", True), ("n", False), ("no way", False)],
    )
    def test_resuming_sets_approved(self, answer, approved):
        graph = approval_graph()
        config = {"configurable": {"thread_id": f"t-{answer}"}}

        graph.invoke({}, config)
        result = graph.invoke(Command(resume=answer), config)

        assert result["approved"] == approved

    def test_flight_graph_compiled_with_checkpointer(self):
        assert build_graph().checkpointer is not None
