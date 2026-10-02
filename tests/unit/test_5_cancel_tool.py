from unittest.mock import MagicMock

from flight_assistant.api import ApiError, api
from flight_assistant.tools import cancel_pnr_tool, interrupt_on, travel_tools


class TestCancelPnrTool:
    """TODO 5 — cancel_pnr tool"""

    def test_named_cancel_pnr_with_description_and_schema(self):
        assert cancel_pnr_tool.name == "cancel_pnr"
        assert len(cancel_pnr_tool.description) > 20
        cancel_pnr_tool.args_schema.model_validate({"pnr_code": "ABC123"})  # should not raise

    def test_given_to_travel_agent(self):
        assert "cancel_pnr" in [t.name for t in travel_tools]

    def test_needs_human_approval(self):
        assert interrupt_on.get("cancel_pnr")
        assert "reject" in interrupt_on["cancel_pnr"]["allowed_decisions"]

    def test_calls_api_cancel_and_returns_result(self, monkeypatch):
        cancel = MagicMock(return_value={"refund": 80, "balance": 180, "pnr": {}})
        monkeypatch.setattr(api, "cancel", cancel)

        output = cancel_pnr_tool.invoke({"pnr_code": "ABC123"})

        cancel.assert_called_once_with("ABC123")
        assert "180" in str(output)

    def test_returns_api_errors_as_text(self, monkeypatch):
        monkeypatch.setattr(api, "cancel", MagicMock(side_effect=ApiError(404, "PNR not found")))

        output = cancel_pnr_tool.invoke({"pnr_code": "NOPE00"})

        assert "PNR not found" in str(output)
