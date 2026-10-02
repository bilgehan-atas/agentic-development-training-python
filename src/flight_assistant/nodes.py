"""
NODES: the steps of the graph.

A node is just a function: (state) -> partial state update (a dict).
LangGraph merges the returned dict into the shared state (see state.py).

This file shows three flavours of node:
  1. Plain code, no LLM                  -> load_context, report_error, ask_clarification, confirm_cancel
  2. A single, direct LLM call           -> classify_intent, answer_info, cancel_booking
     (YOU decide the steps; the LLM fills in one blank)
  3. An agent built with `create_agent`  -> travel_agent
     (the LLM decides the steps: which tools to call, how often, in which order)

`state["messages"]` is the whole conversation of the thread (all earlier turns
too), so pass it to the LLM: then follow-ups like "cancel it" make sense.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel, Field

from .api import ApiError, api, api_url
from .approval import require_approval
from .context import describe_user, today_line
from .llm import llm
from .state import Intent, State
from .tools import interrupt_on, travel_tools


def reply(text: str) -> dict:
    return {"messages": [AIMessage(text)]}


# ─────────────────────────────────────────────────────────────────────────────
# ✅ GIVEN - plain-code node (no LLM at all).
# Fetches the account from the server and puts it into the state.
# ─────────────────────────────────────────────────────────────────────────────
def load_context(state: State) -> dict:
    try:
        # `error: None` clears an error left in the thread by an earlier turn.
        return {"user": api.get_user(), "error": None}
    except Exception as err:  # noqa: BLE001 - surfaced to the user as text
        return {"error": f"Could not load your account from {api_url()} ({err}). Is the server running?"}


# ✅ GIVEN - plain-code node.
def report_error(state: State) -> dict:
    return reply(f"Sorry, something went wrong: {state.get('error')}")


# ✅ GIVEN - plain-code node.
def ask_clarification(state: State) -> dict:
    return reply(
        "Sorry, I'm not sure what you need. I can:\n"
        "  • show your bookings and balance      (\"What are my bookings?\")\n"
        "  • cancel a booking                    (\"Cancel ABC123\")\n"
        "  • search, book or change flights      (\"Move ABC123 to the cheapest flight on Oct 25\")"
    )


# ─────────────────────────────────────────────────────────────────────────────
# ✅ GIVEN - direct LLM call with STRUCTURED OUTPUT.
#
# `llm.with_structured_output(schema)` makes the model answer with JSON that
# matches the schema, and parses it for you. Perfect when the LLM's answer is
# used by CODE (here: by the routing function in edges.py), not by a human.
# ─────────────────────────────────────────────────────────────────────────────
CLASSIFY_PROMPT = """You classify requests sent to an airline assistant. Pick exactly one intent:
- "info":    questions about the user's own account: balance, bookings, PNRs, what they paid, fees.
- "cancel":  cancel an existing booking WITHOUT booking anything else.
- "travel":  anything that needs the flight schedule: search flights or prices, book a new flight,
             change/move/rebook an existing booking to another flight.
- "unclear": greetings, unrelated or ambiguous requests.
Classify the user's LATEST message. Earlier messages are only context
(e.g. "cancel it" right after talking about a booking is "cancel")."""


class IntentOutput(BaseModel):
    intent: Intent


def classify_intent(state: State) -> dict:
    classifier = llm.with_structured_output(IntentOutput)
    result = classifier.invoke([SystemMessage(CLASSIFY_PROMPT), *state["messages"]])
    return {"intent": result.intent}


# ─────────────────────────────────────────────────────────────────────────────
# 🔲 TODO 2 - plain LLM invocation: `llm.invoke(messages)`.
#
# Answer questions about the account ("What's my balance?", "Which bookings
# do I have?") using ONLY the data loaded by `load_context`. No tools needed:
# everything the model needs can be put into the prompt.
#
# Steps:
#   1. Build a SystemMessage that tells the model it is an airline assistant
#      and gives it the account data: `describe_user(state["user"])`.
#      (Adding `today_line()` helps with questions like "my next flight".)
#   2. `llm.invoke([system_message, *state["messages"]])`
#      (the whole conversation, so follow-up questions work) -> returns an AIMessage.
#   3. Return `{"messages": [that_ai_message]}` - the reducer APPENDS it.
#
# 👀 `classify_intent` above does almost the same (but with structured output).
# ✅ Check: pytest tests/unit/test_2_answer_info.py
#    Try:   python -m flight_assistant.main "What is my balance and which bookings do I have?"
# ─────────────────────────────────────────────────────────────────────────────
def answer_info(state: State) -> dict:
    return reply("TODO 2: implement answer_info in src/flight_assistant/nodes.py")


# ─────────────────────────────────────────────────────────────────────────────
# ✅ TODO 3 (solved) - a fixed WORKFLOW of two nodes: the LLM fills in one blank,
# code does the rest.
#
#   cancel_booking  LLM with structured output -> which PNR does the user mean?
#   confirm_cancel  Code -> ask for approval, call api.cancel, format the answer.
#
# Two nodes, because on resume LangGraph re-runs the paused node from its first
# line: the LLM call must not be in the node that calls `require_approval()`,
# or it could pick a different booking than the one the user approved.
#
# Compare with `travel_agent` below, where the LLM decides the steps itself.
# ─────────────────────────────────────────────────────────────────────────────
class PnrOutput(BaseModel):
    pnr_code: str = Field(description="The 6-character booking code to cancel, or an empty string if unclear")


def cancel_booking(state: State) -> dict:
    extractor = llm.with_structured_output(PnrOutput)
    result = extractor.invoke(
        [
            SystemMessage(
                f"Find the booking the user wants to cancel in their latest message. {today_line()}\n"
                f"The user's bookings:\n{describe_user(state.get('user'))}"
            ),
            *state["messages"],
        ]
    )
    return {"pnr_code": result.pnr_code.strip().upper()}


def confirm_cancel(state: State) -> dict:
    code = state.get("pnr_code")
    if not code:
        return reply("Which booking would you like to cancel? Please give me its PNR code.")

    if not require_approval(f"Cancel booking {code}."):
        return reply(f"OK, booking {code} was NOT cancelled.")

    try:
        result = api.cancel(code)
        return reply(f"Booking {code} is cancelled. Refund: {result['refund']} EUR. New balance: {result['balance']} EUR.")
    except ApiError as err:
        return reply(f"Could not cancel {code}: {err.message}.")


# ─────────────────────────────────────────────────────────────────────────────
# 🔲 TODO 5 (part 2 of 2) - an AGENT built with LangChain's `create_agent`.
#
# Do part 1 first: the `cancel_pnr` tool (+ its approval rule) in src/flight_assistant/tools.py.
#
# Handles everything that needs the flight schedule: search, book, change.
# Unlike `cancel_booking`, we do NOT hard-code the steps. `create_agent` runs a loop:
#
#     LLM -> (tool calls -> tool results -> LLM)* -> final answer
#
# For "move ABC123 to the cheapest flight next week" the model itself decides
# to call list_flights, compare prices, call change_booking, then explain.
# (`create_agent` returns a compiled LangGraph graph - a graph used as one node
# inside our graph.)
#
# Steps:
#   1. `agent = create_agent(model=llm, tools=travel_tools, system_prompt=..., middleware=[...])`
#      with `middleware=[HumanInTheLoopMiddleware(interrupt_on=interrupt_on)]`. `interrupt_on`
#      (tools.py) lists the tools that need a human's OK: after the LLM picks its
#      tool calls and BEFORE any of them runs, the middleware asks about each one.
#      (It uses `interrupt()` under the hood, so it needs TODO 4's checkpointer.)
#   2. The system_prompt is the agent's only context. Include:
#        - its role: an airline assistant for IST->FRA flights, plus `today_line()`
#          (it must turn "next Friday" into a date for list_flights)
#        - the user's account: `describe_user(state["user"])` (PNR codes, balance)
#        - rules: never invent flight ids, always get them from list_flights;
#          act right away without asking for confirmation (the approval step
#          does that); finish by saying what was done and the new balance.
#   3. `result = agent.invoke({"messages": state["messages"]})`
#      `result["messages"]` holds the WHOLE loop: user message, AI tool calls,
#      tool results, final AI answer. (Print it once to see the agent think!)
#   4. Return only the final answer: `{"messages": [result["messages"][-1]]}`
#
# 👀 `answer_info` builds a similar prompt; the tools are in src/flight_assistant/tools.py.
# ✅ Check: pytest tests/unit/test_5_travel_agent.py
#    Try:   python -m flight_assistant.main "Move ABC123 to the evening flight on <some date>"
#           python -m flight_assistant.main "Book the cheapest flight between <date> and <date>"
#           python -m flight_assistant.main "Cancel ABC123 and book the evening flight on <some date> instead"
# ─────────────────────────────────────────────────────────────────────────────
def travel_agent(state: State) -> dict:
    return reply("TODO 5: implement travel_agent in src/flight_assistant/nodes.py")
