"""
Thin client for the flight server in ../server (run it with `npm start` there).

Nothing LangChain-specific here - just HTTP calls. Nodes and tools use
these functions to talk to the "real world".
"""

import os
from typing import Any, Literal, Optional, TypedDict

import requests

Flight = TypedDict(
    "Flight",
    {
        "id": str,
        "from": str,
        "to": str,
        "date": str,
        "time": str,
        "basePrice": float,
        "price": Optional[float],
    },
    total=False,
)


class Pnr(TypedDict, total=False):
    code: str
    flightId: str
    paidPrice: float
    cancellationFee: float
    status: Literal["ACTIVE", "CANCELLED"]
    flight: Optional[Flight]


class User(TypedDict, total=False):
    id: str
    balance: float
    pnrs: list[Pnr]


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def api_url() -> str:
    return os.environ.get("FLIGHT_API_URL", "http://localhost:3000")


def _request(method: str, path: str, **kwargs: Any) -> Any:
    res = requests.request(method, f"{api_url()}{path}", timeout=30, **kwargs)
    try:
        data = res.json()
    except ValueError:
        data = {}
    if not res.ok:
        raise ApiError(res.status_code, data.get("error", f"HTTP {res.status_code}"))
    return data


class Api:
    """Mirrors the JS `api` object: one method per endpoint."""

    def get_user(self) -> User:
        return _request("GET", "/user")

    def list_flights(self, from_date: Optional[str] = None, to_date: Optional[str] = None) -> list[Flight]:
        params = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return _request("GET", "/flights", params=params)

    def book(self, flight_id: str) -> dict:
        return _request("POST", "/book", json={"flightId": flight_id})

    def change(self, pnr_code: str, new_flight_id: str) -> dict:
        return _request("POST", "/change", json={"pnrCode": pnr_code, "newFlightId": new_flight_id})

    def cancel(self, pnr_code: str) -> dict:
        return _request("POST", "/cancel", json={"pnrCode": pnr_code})


api = Api()
