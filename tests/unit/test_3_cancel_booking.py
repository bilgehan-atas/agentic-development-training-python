from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from flight_assistant import nodes
from flight_assistant.api import ApiError, api
from helpers import call_node, prompt_text, state_with, state_with_history


class PnrResult(BaseModel):
    pnr_code: str


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch):
    extractor = MagicMock()
    extractor.invoke.return_value = PnrResult(pnr_code=" abc123 ")
    fake = MagicMock()
    fake.with_structured_output.return_value = extractor
    monkeypatch.setattr(nodes, "llm", fake)
    return fake, extractor


@pytest.fixture(autouse=True)
def fake_require_approval(monkeypatch):
    fake = MagicMock(return_value=True)
    monkeypatch.setattr(nodes, "require_approval", fake)
    return fake


class TestCancelBookingPart1:
    """TODO 3 (part 1) — cancel_booking: the LLM picks the PNR"""

    def test_extracts_pnr_and_stores_it(self, fake_llm):
        _, extractor = fake_llm
        update = call_node(nodes.cancel_booking, state_with("Please cancel my October 22 flight"))

        fake_llm[0].with_structured_output.assert_called_once()
        assert "ABC123" in prompt_text(extractor.invoke.call_args[0][0])  # bookings are in the prompt
        assert update["pnr_code"] == "ABC123"  # trimmed + upper-cased

    def test_sends_earlier_conversation_too(self, fake_llm):
        _, extractor = fake_llm
        call_node(nodes.cancel_booking, state_with_history("Cancel it please"))

        prompt = prompt_text(extractor.invoke.call_args[0][0])
        assert "What are my bookings?" in prompt
        assert "Cancel it please" in prompt

    def test_does_not_approve_or_cancel(self, fake_require_approval, monkeypatch):
        cancel = MagicMock()
        monkeypatch.setattr(api, "cancel", cancel)

        call_node(nodes.cancel_booking, state_with("cancel ABC123"))

        fake_require_approval.assert_not_called()
        cancel.assert_not_called()


class TestConfirmCancelPart2:
    """TODO 3 (part 2) — confirm_cancel: ask, then cancel"""

    def test_asks_approval_cancels_and_reports(self, fake_require_approval, monkeypatch):
        cancel = MagicMock(return_value={"refund": 80, "balance": 180, "pnr": {}})
        monkeypatch.setattr(api, "cancel", cancel)

        update = call_node(nodes.confirm_cancel, state_with("cancel my flight", pnr_code="ABC123"))

        fake_require_approval.assert_called_once()
        cancel.assert_called_once_with("ABC123")
        answer = str(update["messages"][0].content)
        assert "80" in answer
        assert "180" in answer

    def test_does_not_call_the_llm(self, fake_llm, monkeypatch):
        monkeypatch.setattr(api, "cancel", MagicMock(return_value={"refund": 80, "balance": 180, "pnr": {}}))

        call_node(nodes.confirm_cancel, state_with("cancel my flight", pnr_code="ABC123"))

        fake_llm[0].invoke.assert_not_called()
        fake_llm[0].with_structured_output.assert_not_called()

    def test_does_not_cancel_when_rejected(self, fake_require_approval, monkeypatch):
        fake_require_approval.return_value = False
        cancel = MagicMock()
        monkeypatch.setattr(api, "cancel", cancel)

        update = call_node(nodes.confirm_cancel, state_with("cancel ABC123", pnr_code="ABC123"))

        cancel.assert_not_called()
        assert len(update["messages"]) == 1

    def test_asks_which_booking_when_no_pnr(self, fake_require_approval, monkeypatch):
        cancel = MagicMock()
        monkeypatch.setattr(api, "cancel", cancel)

        update = call_node(nodes.confirm_cancel, state_with("cancel my flight", pnr_code=""))

        cancel.assert_not_called()
        fake_require_approval.assert_not_called()
        assert "which" in str(update["messages"][0].content).lower()

    def test_reports_api_errors_instead_of_crashing(self, monkeypatch):
        monkeypatch.setattr(api, "cancel", MagicMock(side_effect=ApiError(400, "PNR is not active")))

        update = call_node(nodes.confirm_cancel, state_with("cancel ABC123", pnr_code="ABC123"))

        assert "PNR is not active" in str(update["messages"][0].content)
