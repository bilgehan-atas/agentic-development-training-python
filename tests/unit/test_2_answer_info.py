from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from flight_assistant import nodes
from helpers import call_node, prompt_text, state_with, state_with_history


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch):
    fake = MagicMock()
    fake.invoke.return_value = AIMessage("Your balance is 100 EUR.")
    monkeypatch.setattr(nodes, "llm", fake)
    return fake


class TestAnswerInfo:
    """TODO 2 — answer_info (plain llm.invoke)"""

    def test_calls_llm_invoke_once_and_appends_answer(self, fake_llm):
        update = call_node(nodes.answer_info, state_with("What is my balance?"))

        fake_llm.invoke.assert_called_once()
        assert len(update["messages"]) == 1
        assert "100 EUR" in str(update["messages"][0].content)

    def test_puts_account_data_and_question_into_prompt(self, fake_llm):
        call_node(nodes.answer_info, state_with("What is my balance?"))

        prompt = prompt_text(fake_llm.invoke.call_args[0][0])
        assert "ABC123" in prompt  # from describe_user(state["user"])
        assert "What is my balance?" in prompt

    def test_sends_earlier_conversation_too(self, fake_llm):
        call_node(nodes.answer_info, state_with_history("How much did I pay for it?"))

        prompt = prompt_text(fake_llm.invoke.call_args[0][0])
        assert "What are my bookings?" in prompt
        assert "How much did I pay for it?" in prompt
