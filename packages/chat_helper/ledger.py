"""Month spend, chat turn counts, and unanswered text.

The conditional update is apply_reserve / apply_settle. A retry with the same
key does not reserve again and does not release twice. The TTL attribute is
cleanup only. Reads apply the chat and unanswered windows from the chat policy.

Chat turns (also the live-answer counter) and unanswered text are rows in
CHAT_TABLE when that env var is set. The month spend row is unchanged.
"""
from __future__ import annotations

import copy
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

# DynamoDB TTL attribute. Must be the table's ttl.attribute_name.
# In-repo records already use this name. LevKesha/infrastructure#27 was not
# readable from this environment (private repo), so a different infra name
# would be a mismatch.
TTL_ATTRIBUTE = "ttl"
CHAT_PREFIX = "CHAT#"
UNANSWERED_PREFIX = "UNANSWERED#"



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
    booked = int(actual)
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
                TTL_ATTRIBUTE: int((now + self.chat_ttl).timestamp()),
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
            TTL_ATTRIBUTE: int((now + self.unanswered_ttl).timestamp()),
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
        self._chat_lock = threading.Lock()

    def _load_month(self, month: str) -> dict | None:
        item = self._get(month)
        if not item:
            return None
        item.pop("pk", None)
        return item

    def _view(self, item: dict | None) -> dict | None:
        if item is None:
            return None
        return {"settled": int(item.get("settled") or 0), "holds": item.get("holds") or {}}

    def _commit_month(self, month: str, current: dict | None, updated: dict) -> None:
        """Persist apply_reserve / apply_settle. The version condition is the race."""
        reserved = sum(
            int(hold["amount"]) for hold in updated["holds"].values() if hold["status"] == "reserved"
        )
        seen = 0 if current is None else int(current.get("ver") or 0)
        values: dict[str, Any] = {
            ":settled": int(updated["settled"]),
            ":reserved": reserved,
            ":holds": updated["holds"],
            ":next": seen + 1,
        }
        if current is None:
            condition = "attribute_not_exists(#ver)"
        else:
            condition = "#ver = :seen"
            values[":seen"] = seen
        self.client.update_item(
            TableName=self.table,
            Key={"pk": month},
            ConditionExpression=condition,
            UpdateExpression="SET settled = :settled, reserved = :reserved, #holds = :holds, #ver = :next",
            ExpressionAttributeNames={"#holds": "holds", "#ver": "ver"},
            ExpressionAttributeValues=values,
        )

    def reserve(self, month: str, idem: str, amount: int) -> Reservation:
        for _ in range(8):
            current = self._load_month(month)
            existing = ((current or _blank())["holds"]).get(idem) if current else None
            if existing:
                return Reservation(
                    True,
                    replay=True,
                    settled=existing.get("status") == "settled",
                    response=existing.get("response"),
                )
            updated = apply_reserve(self._view(current), idem, amount, self.cap)
            if updated is None:
                return Reservation(False)
            try:
                self._commit_month(month, current, updated)
            except ConditionalCheckFailed:
                continue
            return Reservation(True)
        raise LedgerError("unavailable")

    def settle(self, month: str, idem: str, actual: int, response: dict) -> Reservation:
        for _ in range(8):
            current = self._load_month(month)
            view = self._view(current)
            existing = (view or _blank())["holds"].get(idem) if view else None
            if existing and existing.get("status") == "settled":
                return Reservation(True, replay=True, settled=True, response=existing.get("response"))
            if view is None or existing is None:
                raise LedgerError("settle condition failed")
            updated = apply_settle(view, idem, actual, response)
            if updated is None:
                raise LedgerError("settle condition failed")
            try:
                self._commit_month(month, current, updated)
            except ConditionalCheckFailed:
                continue
            return Reservation(True, settled=True, response=response)
        raise LedgerError("unavailable")

    def _get(self, pk: str) -> dict | None:
        found = self.client.get_item(TableName=self.table, Key={"pk": pk})
        item = found.get("Item")
        return item if item else None

    def _put(self, item: dict) -> None:
        self.client.put_item(TableName=self.table, Item=item)

    def _scan_prefix(self, prefix: str) -> list[dict]:
        items: list[dict] = []
        kwargs: dict[str, Any] = {
            "TableName": self.table,
            "FilterExpression": "begins_with(pk, :prefix)",
            "ExpressionAttributeValues": {":prefix": prefix},
        }
        while True:
            found = self.client.scan(**kwargs)
            items.extend(found.get("Items") or [])
            last = found.get("LastEvaluatedKey")
            if not last:
                return items
            kwargs["ExclusiveStartKey"] = last

    def _open_chat(self, chat_id: str, now: datetime) -> dict | None:
        item = self._get(CHAT_PREFIX + chat_id)
        if item is None:
            return None
        created = datetime.fromisoformat(str(item["created"]))
        if now - created >= self.chat_ttl:
            return None
        return item

    def read_turns(self, chat_id: str, now: datetime) -> int:
        item = self._open_chat(chat_id, now)
        if item is None:
            return 0
        return int(item["turns"])

    def chat_closed(self, chat_id: str, now: datetime) -> bool:
        item = self._open_chat(chat_id, now)
        if item is None:
            return False
        return bool(item.get("closed"))

    def write_chat(self, chat_id: str, turns: int, now: datetime, closed: bool = False) -> None:
        with self._chat_lock:
            item = self._open_chat(chat_id, now)
            if item is None:
                item = {
                    "pk": CHAT_PREFIX + chat_id,
                    "created": now.isoformat(),
                    "turns": 0,
                    "closed": False,
                    TTL_ATTRIBUTE: int((now + self.chat_ttl).timestamp()),
                }
            item["pk"] = CHAT_PREFIX + chat_id
            item["turns"] = int(turns)
            item["closed"] = bool(item.get("closed") or closed)
            self._put(item)

    def bump(self, chat_id: str, now: datetime) -> int:
        turns = self.read_turns(chat_id, now) + 1
        self.write_chat(chat_id, turns, now)
        return turns

    def mark_closed(self, chat_id: str, now: datetime) -> None:
        self.write_chat(chat_id, self.read_turns(chat_id, now), now, closed=True)

    def write_unanswered(self, text: str, now: datetime) -> dict:
        record = {
            "pk": UNANSWERED_PREFIX + uuid.uuid4().hex,
            "text": text,
            "created": now.isoformat(),
            TTL_ATTRIBUTE: int((now + self.unanswered_ttl).timestamp()),
        }
        self._put(record)
        return {"text": text, "created": record["created"], TTL_ATTRIBUTE: record[TTL_ATTRIBUTE]}

    def read_unanswered(self, now: datetime) -> list[str]:
        kept: list[tuple[str, str]] = []
        for record in self._scan_prefix(UNANSWERED_PREFIX):
            created = datetime.fromisoformat(str(record["created"]))
            if now - created < self.unanswered_ttl:
                kept.append((record["created"], record["text"]))
        kept.sort(key=lambda pair: pair[0])
        return [text for _created, text in kept]


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


class NativeDynamo:
    """Low-level DynamoDB client with plain Python values.

    TableLedger and FakeTable share those shapes. The TTL attribute is written
    as an integer epoch so the table's TTL setting can delete the row.
    """

    def __init__(self, client: Any) -> None:
        from boto3.dynamodb.types import TypeDeserializer, TypeSerializer

        self.client = client
        self._ser = TypeSerializer()
        self._de = TypeDeserializer()

    def _values(self, values: dict) -> dict:
        return {key: self._ser.serialize(self._number(value)) for key, value in values.items()}

    def _number(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._number(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._number(item) for item in value]
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return value
        if isinstance(value, int):
            return value
        # DynamoDB numbers come back as Decimal. Store them again as ints when whole.
        try:
            whole = int(value)
        except (TypeError, ValueError):
            return value
        if whole == value:
            return whole
        return value

    def _item(self, item: dict) -> dict:
        return {key: self._number(self._de.deserialize(value)) for key, value in item.items()}

    def put_item(self, **kwargs: Any) -> dict:
        payload = dict(kwargs)
        payload["Item"] = self._values(payload["Item"])
        return self.client.put_item(**payload)

    def get_item(self, **kwargs: Any) -> dict:
        payload = dict(kwargs)
        payload["Key"] = self._values(payload["Key"])
        found = self.client.get_item(**payload)
        if "Item" not in found:
            return {}
        return {"Item": self._item(found["Item"])}

    def scan(self, **kwargs: Any) -> dict:
        payload = dict(kwargs)
        if "ExpressionAttributeValues" in payload:
            payload["ExpressionAttributeValues"] = self._values(payload["ExpressionAttributeValues"])
        if "ExclusiveStartKey" in payload:
            payload["ExclusiveStartKey"] = self._values(payload["ExclusiveStartKey"])
        found = self.client.scan(**payload)
        out: dict[str, Any] = {"Items": [self._item(item) for item in found.get("Items", [])]}
        if "LastEvaluatedKey" in found:
            out["LastEvaluatedKey"] = self._item(found["LastEvaluatedKey"])
        return out

    def update_item(self, **kwargs: Any) -> dict:
        from botocore.exceptions import ClientError

        payload = dict(kwargs)
        payload["Key"] = self._values(payload["Key"])
        payload["ExpressionAttributeValues"] = self._values(payload["ExpressionAttributeValues"])
        try:
            return self.client.update_item(**payload)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code == "ConditionalCheckFailedException":
                raise ConditionalCheckFailed from exc
            raise


def ledger_from_env(
    env: dict,
    policy: dict,
    client_factory: Callable[[dict], Any] | None = None,
) -> MemoryLedger | TableLedger:
    table = env.get("CHAT_TABLE")
    if not table:
        return MemoryLedger(policy)
    factory = client_factory or dynamo_client_from_env
    return TableLedger(table, NativeDynamo(factory(env)), policy)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
