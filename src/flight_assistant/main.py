"""
Entry point.

    python -m flight_assistant.main "What are my bookings?"   # one question
    python -m flight_assistant.main                            # interactive chat
"""

import sys
from uuid import uuid4

from .graph import build_graph
from .llm import MODEL
from .run import run_turn


def approve(question: str) -> str:
    return input(f"\n  ⚠️  {question} ")


def ask(app, thread_id: str, question: str) -> None:
    state = run_turn(app, question, thread_id=thread_id, approve=approve)
    print(f"\n🤖 {state['messages'][-1].content}\n")


def main() -> None:
    app = build_graph()
    # One thread for the whole session: the checkpointer (TODO 4) keeps its state,
    # so every turn sees the earlier conversation. A new process = a new thread.
    thread_id = str(uuid4())
    question = " ".join(sys.argv[1:]).strip()

    if question:
        ask(app, thread_id, question)
    else:
        print(f'✈️  Flight assistant (model: {MODEL}). Type "exit" to quit.\n')
        while True:
            line = input("🧑 ").strip()
            if not line or line == "exit":
                break
            ask(app, thread_id, line)


if __name__ == "__main__":
    main()
