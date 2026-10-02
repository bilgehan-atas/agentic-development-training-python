from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from flight_assistant import nodes
from flight_assistant.tools import travel_tools
from helpers import call_node, state_with, state_with_history


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch):
    fake = MagicMock()
    monkeypatch.setattr(nodes, "llm", fake)
    return fake


@pytest.fixture(autouse=True)
def fake_create_agent(monkeypatch):
    fake_agent = MagicMock()
    # What a real agent run returns: the whole loop, final answer last.
    fake_agent.invoke.return_value = {
        "messages": [
            HumanMessage("Book the cheapest flight on 2026-10-14"),
            AIMessage(content="", tool_calls=[{"id": "1", "name": "list_flights", "args": {}}]),
            ToolMessage(tool_call_id="1", content="[...]"),
            AIMessage("Booked IST-FRA-20261014-0800. New balance: 0 EUR."),
        ]
    }
    create_agent = MagicMock(return_value=fake_agent)
    monkeypatch.setattr(nodes, "create_agent", create_agent)
    return create_agent, fake_agent


@pytest.fixture(autouse=True)
def fake_hitl_middleware(monkeypatch):
    middleware_instance = object()
    factory = MagicMock(return_value=middleware_instance)
    monkeypatch.setattr(nodes, "HumanInTheLoopMiddleware", factory)
    return factory, middleware_instance


def agent_kwargs(create_agent):
    return create_agent.call_args.kwargs


class TestTravelAgent:
    """TODO 5 — travel_agent (create_agent)"""

    def test_creates_agent_with_local_llm_and_travel_tools(self, fake_llm, fake_create_agent):
        create_agent, _ = fake_create_agent
        call_node(nodes.travel_agent, state_with("Book the cheapest flight on 2026-10-14"))

        create_agent.assert_called_once()
        kwargs = agent_kwargs(create_agent)
        assert kwargs["model"] is fake_llm
        assert kwargs["tools"] is travel_tools

    def test_asks_human_before_sensitive_tools(self, fake_create_agent, fake_hitl_middleware):
        create_agent, _ = fake_create_agent
        factory, middleware_instance = fake_hitl_middleware

        call_node(nodes.travel_agent, state_with("Book the cheapest flight on 2026-10-14"))

        factory.assert_called_once()
        interrupt_on = factory.call_args.kwargs["interrupt_on"]
        for name in ["book_flight", "change_booking", "cancel_pnr"]:
            assert name in interrupt_on
        assert "list_flights" not in interrupt_on
        assert middleware_instance in agent_kwargs(create_agent)["middleware"]

    def test_gives_agent_account_data_and_today(self, fake_create_agent):
        create_agent, _ = fake_create_agent
        call_node(nodes.travel_agent, state_with("Book the cheapest flight on 2026-10-14"))

        system_prompt = agent_kwargs(create_agent)["system_prompt"]
        assert "ABC123" in system_prompt  # describe_user(state["user"])
        assert "Today is" in system_prompt  # today_line()

    def test_invokes_agent_with_conversation(self, fake_create_agent):
        _, fake_agent = fake_create_agent
        call_node(nodes.travel_agent, state_with_history("Move it to the evening flight"))

        contents = [m.content for m in fake_agent.invoke.call_args[0][0]["messages"]]
        assert "What are my bookings?" in contents
        assert "Move it to the evening flight" in contents

    def test_returns_only_final_answer(self):
        update = call_node(nodes.travel_agent, state_with("Book the cheapest flight on 2026-10-14"))

        assert len(update["messages"]) == 1
        assert "New balance: 0 EUR" in str(update["messages"][0].content)
