from langchain_core.messages import AIMessage, HumanMessage

from flight_assistant.api import User
from flight_assistant.state import State

sample_user: User = {
    "id": "user1",
    "balance": 100,
    "pnrs": [
        {
            "code": "ABC123",
            "flightId": "IST-FRA-20261022-0800",
            "paidPrice": 100,
            "cancellationFee": 20,
            "status": "ACTIVE",
            "flight": {
                "id": "IST-FRA-20261022-0800",
                "from": "IST",
                "to": "FRA",
                "date": "2026-10-22",
                "time": "08:00",
                "basePrice": 100,
            },
        }
    ],
}


def state_with(question: str, **extra) -> State:
    state: State = {"messages": [HumanMessage(question)], "user": sample_user}
    state.update(extra)
    return state


def state_with_history(question: str) -> State:
    """A state whose thread already has an earlier turn ("What are my bookings?" -> answer)."""
    return state_with(
        question,
        messages=[
            HumanMessage("What are my bookings?"),
            AIMessage("You have one booking: ABC123 on 2026-10-22."),
            HumanMessage(question),
        ],
    )


def call_node(node, state: State) -> dict:
    """Call a node function directly, outside of a graph."""
    return node(state)


def prompt_text(messages) -> str:
    """Text of all messages passed to a (mocked) LLM call."""
    return "\n".join(str(m.content) for m in messages)
