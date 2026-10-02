from flight_assistant.graph import build_graph


def test_classify_intent_has_conditional_edge_to_each_intent_node():
    """TODO 1 — graph wiring"""
    graph = build_graph().get_graph()
    targets = [(e.target, e.conditional) for e in graph.edges if e.source == "classify_intent"]

    for target in ["answer_info", "cancel_booking", "travel_agent", "ask_clarification"]:
        assert (target, True) in targets


def test_cancel_booking_to_confirm_cancel_to_end_others_to_end():
    graph = build_graph().get_graph()
    edges = [f"{e.source}->{e.target}" for e in graph.edges]

    assert "cancel_booking->confirm_cancel" in edges
    for node in ["answer_info", "confirm_cancel", "travel_agent"]:
        assert f"{node}->__end__" in edges
