"""
End-to-end tests: real Ollama model + a fresh flight server started by the test.
Run with `RUN_INTEGRATION_TESTS=1 pytest tests/integration` (Ollama must be running with the model pulled).
"""

import os
import subprocess
import time
import uuid
from datetime import date, timedelta
from pathlib import Path

import pytest

PORT = 3999
os.environ["FLIGHT_API_URL"] = f"http://localhost:{PORT}"

from flight_assistant.api import api  # noqa: E402
from flight_assistant.graph import build_graph  # noqa: E402
from flight_assistant.run import run_turn  # noqa: E402

SERVER_DIR = Path(__file__).resolve().parents[2].parent / "server"

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION_TESTS"),
    reason="set RUN_INTEGRATION_TESTS=1 to run (needs Ollama + node)",
)


def ask(question: str) -> dict:
    state = run_turn(
        build_graph(),
        question,
        thread_id=str(uuid.uuid4()),
        approve=lambda _: "y",
        log=lambda _: None,
    )
    return {"state": state, "answer": str(state["messages"][-1].content)}


def days_from_now(n: int) -> str:
    return (date.today() + timedelta(days=n)).isoformat()


@pytest.fixture(scope="module")
def server():
    proc = subprocess.Popen(
        ["node", "server.js"],
        cwd=SERVER_DIR,
        env={**os.environ, "PORT": str(PORT)},
    )
    for _ in range(50):
        try:
            api.get_user()
            break
        except Exception:
            time.sleep(0.1)
    else:
        proc.kill()
        raise RuntimeError(f"Flight server did not start in {SERVER_DIR}")
    yield proc
    proc.kill()


def test_unclear_request_routes_to_ask_clarification(server):
    result = ask("Hello! How are you?")
    assert result["state"]["intent"] == "unclear"


def test_answers_account_questions_from_loaded_data(server):
    """TODO 2"""
    result = ask("What is my current balance?")
    assert result["state"]["intent"] == "info"
    assert "100" in result["answer"]


def test_cancels_booking_through_cancel_booking_node(server):
    """TODO 3"""
    result = ask("Please cancel my booking ABC123")
    assert result["state"]["intent"] == "cancel"

    user = api.get_user()
    pnr = next(p for p in user["pnrs"] if p["code"] == "ABC123")
    assert pnr["status"] == "CANCELLED"
    assert user["balance"] == 180


def test_travel_agent_books_cheapest_flight_in_date_range(server):
    """travel agent — multi-step tool use"""
    from_date = days_from_now(20)
    to_date = days_from_now(23)
    cheapest = min(f["price"] for f in api.list_flights(from_date, to_date))

    result = ask(f"Book the cheapest flight between {from_date} and {to_date}.")
    assert result["state"]["intent"] == "travel"

    booked = [p for p in api.get_user()["pnrs"] if p["status"] == "ACTIVE"]
    assert len(booked) == 1
    assert booked[0]["paidPrice"] == cheapest
