"""
TOOLS for the travel agent (see `travel_agent` in nodes.py).

A tool = a function + a name + a description + an input schema.
The LLM never runs code itself: it only reads the name/description/schema
and replies "please call list_flights with {...}". `create_agent` then
runs the function and feeds the result back to the LLM.

=> Write descriptions for the LLM, not for humans!

The tools that move money do NOT ask for approval themselves: the agent's
`HumanInTheLoopMiddleware` pauses BEFORE running them (see `interrupt_on` below).
"""

import json
from typing import Callable

from langchain.tools import tool
from pydantic import BaseModel, Field

from .api import ApiError, api
from .context import weekday_of


def run(action: Callable[[], object]) -> str:
    """Tools return text to the LLM - including errors, so it can react to them."""
    try:
        return json.dumps(action())
    except ApiError as err:
        return f"Error {err.status}: {err.message}"


# ✅ GIVEN
class ListFlightsInput(BaseModel):
    start_date: str = Field(description="First date, YYYY-MM-DD")
    end_date: str = Field(description="Last date, YYYY-MM-DD")


@tool("list_flights", args_schema=ListFlightsInput)
def list_flights_tool(start_date: str, end_date: str) -> str:
    """List IST->FRA flights with their current price (EUR) between two dates (inclusive).
    Use it to find flight ids and compare prices. Keep the range small (max ~14 days)."""

    def action():
        flights = api.list_flights(start_date, end_date)
        return [
            {"id": f["id"], "day": weekday_of(f["date"]), "date": f["date"], "time": f["time"], "price": f.get("price")}
            for f in flights
        ]

    return run(action)


# ✅ GIVEN
class BookFlightInput(BaseModel):
    flight_id: str = Field(description="Flight id from list_flights, e.g. IST-FRA-20261015-0800")


@tool("book_flight", args_schema=BookFlightInput)
def book_flight_tool(flight_id: str) -> str:
    """Book a NEW ticket on a flight. The current price is charged from the user's balance.
    Do NOT use this to move an existing booking - use change_booking for that."""
    return run(lambda: api.book(flight_id))


# ✅ GIVEN
class ChangeBookingInput(BaseModel):
    pnr_code: str = Field(description="6-character booking code, e.g. ABC123")
    new_flight_id: str = Field(description="Flight id from list_flights")


@tool("change_booking", args_schema=ChangeBookingInput)
def change_booking_tool(pnr_code: str, new_flight_id: str) -> str:
    """Move an existing ACTIVE booking (PNR) to another flight. The old ticket is refunded
    (paid price minus cancellation fee) and the new flight's current price is charged."""
    return run(lambda: api.change(pnr_code, new_flight_id))


# ✅ TODO 5 (part 1 of 2, solved)
class CancelPnrInput(BaseModel):
    pnr_code: str = Field(description="6-character booking code to cancel, e.g. ABC123")


@tool("cancel_pnr", args_schema=CancelPnrInput)
def cancel_pnr_tool(pnr_code: str) -> str:
    """Cancel an existing ACTIVE booking (PNR), refunding it (paid price minus cancellation
    fee) to the user's balance. Use this instead of change_booking when the user wants to
    drop a booking entirely rather than move it to another flight."""
    return run(lambda: api.cancel(pnr_code))


travel_tools = [list_flights_tool, book_flight_tool, change_booking_tool, cancel_pnr_tool]

# ✅ GIVEN (except the cancel_pnr rule - TODO 5) - which tool calls need a human's OK.
#
# Passed to `HumanInTheLoopMiddleware(interrupt_on=interrupt_on)` in `travel_agent`. When
# the LLM asks for one of these tools, the middleware calls `interrupt()` BEFORE the
# tool runs, with ALL such tool calls of that step in one request, so every
# action is shown and approved (or rejected) separately. Tools not listed here
# (list_flights) run without asking. `description` is the question shown to the user.
APPROVE_OR_REJECT = ["approve", "reject"]

interrupt_on: dict = {
    "book_flight": {
        "allowed_decisions": APPROVE_OR_REJECT,
        "description": lambda call, state, runtime: f"Book a NEW ticket on flight {call['args']['flight_id']}.",
    },
    "change_booking": {
        "allowed_decisions": APPROVE_OR_REJECT,
        "description": lambda call, state, runtime: (
            f"Move booking {call['args']['pnr_code']} to flight {call['args']['new_flight_id']}."
        ),
    },
    "cancel_pnr": {
        "allowed_decisions": APPROVE_OR_REJECT,
        "description": lambda call, state, runtime: f"Cancel booking {call['args']['pnr_code']}.",
    },
}
