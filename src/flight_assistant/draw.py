"""Prints the graph as a Mermaid diagram: `python -m flight_assistant.draw` (paste into https://mermaid.live)."""

from .graph import build_graph

if __name__ == "__main__":
    print(build_graph().get_graph().draw_mermaid())
