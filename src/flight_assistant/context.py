"""
Helpers that turn raw data into text an LLM can read.
("Context engineering": the model only knows what you put in the prompt!)
"""

from datetime import date, datetime
from typing import Optional

from .api import User

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def to_iso_date(d: date) -> str:
    return d.isoformat()


def weekday_of(iso_date: str) -> str:
    return WEEKDAYS[datetime.fromisoformat(iso_date).weekday()]


def today_line(now: Optional[date] = None) -> str:
    """e.g. "Today is Wednesday 2026-09-30." - LLMs don't know the current date."""
    now = now or date.today()
    iso = to_iso_date(now)
    return f"Today is {weekday_of(iso)} {iso}."


def describe_user(user: Optional[User]) -> str:
    """Human/LLM-readable summary of the user's account."""
    if not user:
        return "No account data available."
    lines = [f"Balance: {user['balance']} EUR", "Bookings (PNRs):"]
    pnrs = user.get("pnrs", [])
    if not pnrs:
        lines.append("  (none)")
    for p in pnrs:
        f = p.get("flight")
        if f:
            when = f"{f['from']}->{f['to']} on {weekday_of(f['date'])} {f['date']} at {f['time']}"
        else:
            when = p["flightId"]
        lines.append(
            f"  - {p['code']} [{p['status']}] {when} (flightId {p['flightId']}), "
            f"paid {p['paidPrice']} EUR, cancellation fee {p['cancellationFee']} EUR"
        )
    return "\n".join(lines)
