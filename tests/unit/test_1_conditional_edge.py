import pytest

from flight_assistant.edges import route_after_load, route_by_intent


class TestRouteAfterLoad:
    """route_after_load (given)"""

    def test_goes_to_classify_intent_when_loaded(self):
        assert route_after_load({}) == "classify_intent"

    def test_goes_to_report_error_when_loading_failed(self):
        assert route_after_load({"error": "server down"}) == "report_error"


@pytest.mark.parametrize(
    "intent,node",
    [
        ("info", "answer_info"),
        ("cancel", "cancel_booking"),
        ("travel", "travel_agent"),
        ("unclear", "ask_clarification"),
    ],
)
def test_routes_intent_to_node(intent, node):
    """TODO 1 — route_by_intent"""
    assert route_by_intent({"intent": intent}) == node


def test_falls_back_to_ask_clarification_without_intent():
    assert route_by_intent({}) == "ask_clarification"
