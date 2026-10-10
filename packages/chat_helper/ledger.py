"""Month spend, chat turn counts, and unanswered text.

The conditional update is apply_reserve / apply_settle. A retry with the same
key does not reserve again and does not release twice. TTL fields are cleanup
only. Reads apply the chat and unanswered windows from the chat policy.
"""
from __future__ import annotations

import copy
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable



class LedgerError(Exception):
    """The ledger cannot be updated. Callers must not call the model."""


class ConditionalCheckFailed(Exception):
    pass


@dataclass
class Reservation:
    ok: bool
    replay: bool = False
    settled: bool = False
    response: dict | None = None


def month_key(now: datetime) -> str:
    return f"SPEND#{now.strftime('%Y-%m')}"


def _blank() -> dict[str, Any]:
    return {"settled": 0, "holds": {}}


def apply_reserve(item: dict | None, idem: str, amount: int, cap: int) -> dict | None:
    """Return the next item, or None when the condition fails."""
    current = item or _blank()
    holds = current["holds"]
    if idem in holds:
        return None
    reserved = sum(hold["amount"] for hold in holds.values() if hold["status"] == "reserved")
    if int(current["settled"]) + reserved + amount > cap:
        return None
    updated = copy.deepcopy(current)
    updated["holds"][idem] = {"amount": amount, "status": "reserved"}
    return updated


def apply_settle(item: dict, idem: str, actual: int, response: dict) -> dict | None:
    hold = item["holds"].get(idem)
    if hold is None or hold["status"] != "reserved":
        return None
    updated = copy.deepcopy(item)
    booked = min(int(actual), int(hold["amount"]))
    updated["settled"] = int(updated["settled"]) + booked
    updated["holds"][idem] = {
        "amount": hold["amount"],
        "status": "settled",
        "actual": booked,
        "response": response,
    }
    return updated


class MemoryLedger:
    def __init__(self, policy: dict) -> None:
        self.cap = int(policy["cap_micro"])
        self.chat_ttl = timedelta(hours=int(policy["chat_ttl_hours"]))
        self.unanswered_ttl = timedelta(days=int(policy["unanswered_ttl_days"]))
        self._lock = threading.Lock()
        self.months: dict[str, dict] = {}
        self.chats: dict[str, dict] = {}
        self.unanswered: list[dict] = []
        self.down = False

    def reserve(self, month: str, idem: str, amount: int) -> Reservation:
        if self.down:
            raise LedgerError("unavailable")
        with self._lock:
            current = self.months.get(month)
            existing = (current or _blank())["holds"].get(idem)
            if existing:
                return Reservation(
                    True,
                    replay=True,
                    settled=existing["status"] == "settled",
                    response=existing.get("response"),
                )
            updated = apply_reserve(current, idem, amount, self.cap)
            if updated is None:
                return Reservation(False)
            self.months[month] = updated
            return Reservation(True)

    def settle(self, month: str, idem: str, actual: int, response: dict) -> Reservation:
        if self.down:
            raise LedgerError("unavailable")
        with self._lock:
            current = self.months[month]
            existing = current["holds"][idem]
            if existing["status"] == "settled":
                return Reservation(True, replay=True, settled=True, response=existing.get("response"))
            updated = apply_settle(current, idem, actual, response)
            if updated is None:
                raise LedgerError("settle condition failed")
            self.months[month] = updated
            return Reservation(True, settled=True, response=response)

    def read_turns(self, chat_id: str, now: datetime) -> int:
        item = self.chats.get(chat_id)
        if item is None:
            return 0
        if now - item["created"] >= self.chat_ttl:
            return 0
        return int(item["turns"])

    def chat_closed(self, chat_id: str, now: datetime) -> bool:
        item = self.chats.get(chat_id)
        if item is None or now - item["created"] >= self.chat_ttl:
            return False
        return bool(item.get("closed"))

    def write_chat(self, chat_id: str, turns: int, now: datetime, closed: bool = False) -> None:
        item = self.chats.get(chat_id)
        if item is None or now - item["created"] >= self.chat_ttl:
            item = {
                "created": now,
                "turns": 0,
                "closed": False,
                "ttl": int((now + self.chat_ttl).timestamp()),
            }
            self.chats[chat_id] = item
        item["turns"] = turns
        item["closed"] = bool(item["closed"] or closed)

    def bump(self, chat_id: str, now: datetime) -> int:
        turns = self.read_turns(chat_id, now) + 1
        self.write_chat(chat_id, turns, now)
        return turns

    def mark_closed(self, chat_id: str, now: datetime) -> None:
        self.write_chat(chat_id, self.read_turns(chat_id, now), now, closed=True)

    def write_unanswered(self, text: str, now: datetime) -> dict:
        record = {
            "text": text,
            "created": now.isoformat(),
            "ttl": int((now + self.unanswered_ttl).timestamp()),
        }
        self.unanswered.append(record)
        return record

    def read_unanswered(self, now: datetime) -> list[str]:
        kept = []
        for record in self.unanswered:
            created = datetime.fromisoformat(record["created"])
            if now - created < self.unanswered_ttl:
                kept.append(record["text"])
        return kept


class TableLedger:
    """Same conditions as MemoryLedger, through a conditional update client."""

    def __init__(self, table: str, client: Any, policy: dict) -> None:
        self.table = table
        self.client = client
        self.cap = int(policy["cap_micro"])
        self.chat_ttl = timedelta(hours=int(policy["chat_ttl_hours"]))
        self.unanswered_ttl = timedelta(days=int(policy["unanswered_ttl_days"]))
        self.chats: dict[str, dict] = {}
        self.unanswered: list[dict] = []

    def reserve(self, month: str, idem: str, amount: int) -> Reservation:
        try:
            self.client.update_item(
                TableName=self.table,
                Key={"pk": month},
                ConditionExpression=(
                    "attribute_not_exists(#holds.#idem) AND "
                    "if_not_exists(settled, :zero) + if_not_exists(reserved, :zero) + :amt <= :cap"
                ),
                UpdateExpression=(
                    "SET settled = if_not_exists(settled, :zero), "
                    "reserved = if_not_exists(reserved, :zero) + :amt, "
                    "#holds.#idem = :hold"
                ),
                ExpressionAttributeNames={"#holds": "holds", "#idem": idem},
                ExpressionAttributeValues={
                    ":amt": amount,
                    ":zero": 0,
                    ":cap": self.cap,
                    ":hold": {"amount": amount, "status": "reserved"},
                },
            )
        except ConditionalCheckFailed:
            item = self.client.get_item(TableName=self.table, Key={"pk": month}).get("Item") or {}
            hold = (item.get("holds") or {}).get(idem)
            if hold:
                return Reservation(
                    True,
                    replay=True,
                    settled=hold.get("status") == "settled",
                    response=hold.get("response"),
                )
            return Reservation(False)
        except LedgerError:
            raise
        return Reservation(True)

    def settle(self, month: str, idem: str, actual: int, response: dict) -> Reservation:
        try:
            self.client.update_item(
                TableName=self.table,
                Key={"pk": month},
                ConditionExpression="#holds.#idem.#status = :reserved",
                UpdateExpression=(
                    "SET settled = settled + :actual, "
                    "reserved = reserved - #holds.#idem.#amount, "
                    "#holds.#idem.#status = :settled, "
                    "#holds.#idem.#actual = :actual, "
                    "#holds.#idem.#response = :response"
                ),
                ExpressionAttributeNames={
                    "#holds": "holds",
                    "#idem": idem,
                    "#status": "status",
                    "#amount": "amount",
                    "#actual": "actual",
                    "#response": "response",
                },
                ExpressionAttributeValues={
                    ":reserved": "reserved",
                    ":settled": "settled",
                    ":actual": actual,
                    ":response": response,
                },
            )
        except ConditionalCheckFailed:
            item = self.client.get_item(TableName=self.table, Key={"pk": month}).get("Item") or {}
            hold = (item.get("holds") or {}).get(idem) or {}
            if hold.get("status") == "settled":
                return Reservation(True, replay=True, settled=True, response=hold.get("response"))
            raise LedgerError("settle condition failed") from None
        return Reservation(True, settled=True, response=response)

    def read_turns(self, chat_id: str, now: datetime) -> int:
        return MemoryLedger.read_turns(self, chat_id, now)  # type: ignore[arg-type]

    def chat_closed(self, chat_id: str, now: datetime) -> bool:
        return MemoryLedger.chat_closed(self, chat_id, now)  # type: ignore[arg-type]

    def write_chat(self, chat_id: str, turns: int, now: datetime, closed: bool = False) -> None:
        MemoryLedger.write_chat(self, chat_id, turns, now, closed)  # type: ignore[arg-type]

    def bump(self, chat_id: str, now: datetime) -> int:
        return MemoryLedger.bump(self, chat_id, now)  # type: ignore[arg-type]

    def mark_closed(self, chat_id: str, now: datetime) -> None:
        MemoryLedger.mark_closed(self, chat_id, now)  # type: ignore[arg-type]

    def write_unanswered(self, text: str, now: datetime) -> dict:
        return MemoryLedger.write_unanswered(self, text, now)  # type: ignore[arg-type]

    def read_unanswered(self, now: datetime) -> list[str]:
        return MemoryLedger.read_unanswered(self, now)  # type: ignore[arg-type]


class FakeTable:
    """In-process table that applies the same pure conditions under a lock."""

    def __init__(self) -> None:
        self.items: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.updates: list[dict] = []
        self.fail_next = False

    def update_item(self, **kwargs: Any) -> dict:
        self.updates.append(kwargs)
        if self.fail_next:
            self.fail_next = False
            raise LedgerError("unavailable")
        key = kwargs["Key"]["pk"]
        with self.lock:
            current = copy.deepcopy(self.items.get(key) or _blank())
            names = kwargs.get("ExpressionAttributeNames") or {}
            idem = names.get("#idem")
            values = kwargs["ExpressionAttributeValues"]
            if "reserved" in kwargs["ConditionExpression"] and ":amt" in values:
                updated = apply_reserve(current, idem, int(values[":amt"]), int(values[":cap"]))
                if updated is None:
                    raise ConditionalCheckFailed()
                reserved = sum(
                    hold["amount"] for hold in updated["holds"].values() if hold["status"] == "reserved"
                )
                updated["reserved"] = reserved
                self.items[key] = updated
                return {}
            updated = apply_settle(current, idem, int(values[":actual"]), values[":response"])
            if updated is None:
                raise ConditionalCheckFailed()
            reserved = sum(
                hold["amount"] for hold in updated["holds"].values() if hold["status"] == "reserved"
            )
            updated["reserved"] = reserved
            self.items[key] = updated
            return {}

    def get_item(self, **kwargs: Any) -> dict:
        key = kwargs["Key"]["pk"]
        item = self.items.get(key)
        return {"Item": copy.deepcopy(item)} if item else {}


def dynamo_client_from_env(env: dict) -> Any:
    """Production client. Region comes from the environment, with no default."""
    region = env.get("AWS_REGION")
    if not region:
        raise LedgerError("AWS_REGION is unset")
    import boto3

    return boto3.client("dynamodb", region_name=region)


def ledger_from_env(
    env: dict,
    policy: dict,
    client_factory: Callable[[dict], Any] | None = None,
) -> MemoryLedger | TableLedger:
    table = env.get("CHAT_TABLE")
    if not table:
        return MemoryLedger(policy)
    factory = client_factory or dynamo_client_from_env
    return TableLedger(table, factory(env), policy)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
