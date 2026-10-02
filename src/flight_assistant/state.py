"""
Graph STATE: the shared "memory" that flows through every node.

- Each node receives the current state and returns a PARTIAL update (a dict).
- `messages` uses a reducer (`add_messages`) -> returned messages are APPENDED.
- Every other field has no reducer -> a returned value simply OVERWRITES it.

The state lives in a THREAD (`thread_id`). In the interactive chat every turn
reuses the same thread, so `messages` holds the whole conversation and the
other fields keep their value from the previous turn until a node overwrites them.
"""

from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from .api import User

Intent = Literal["info", "cancel", "travel", "unclear"]


class State(TypedDict, total=False):
    # Conversation: the user's requests + the assistant's final answers.
    messages: Annotated[list[BaseMessage], add_messages]
    # Account data loaded from GET /user by `load_context`.
    user: Optional[User]
    # What the user wants, decided by `classify_intent`.
    intent: Optional[Intent]
    # The booking `cancel_booking` picked; `confirm_cancel` asks for approval and cancels it.
    pnr_code: Optional[str]
    # Set by `load_context` when the flight server can't be reached.
    error: Optional[str]
