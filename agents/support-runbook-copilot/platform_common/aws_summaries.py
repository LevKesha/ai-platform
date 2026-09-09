"""AWS text summaries for RAG ingest and MCP tools/resources."""
from __future__ import annotations

import logging

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)


def fetch_ecs_summary(region: str) -> str:
    try:
        client = boto3.client("ecs", region_name=region)
        response = client.list_clusters()
        arns = response.get("clusterArns", [])
        if not arns:
            return "No ECS clusters found in this account."
        desc = client.describe_clusters(clusters=arns, include=["STATISTICS"])
        lines = ["# ECS clusters\n"]
        for c in desc.get("clusters", []):
            name = c.get("clusterName", "?")
            status = c.get("status", "?")
            running = c.get("runningTasksCount", 0)
            pending = c.get("pendingTasksCount", 0)
            lines.append(
                f"- {name}: status={status}, runningTasks={running}, pendingTasks={pending}"
            )
        return "\n".join(lines)
    except (BotoCoreError, ClientError) as e:
        logger.warning("fetch_ecs_summary failed: %s", e)
        return f"Error fetching ECS clusters: {e}"


def fetch_s3_summary(region: str) -> str:
    try:
        client = boto3.client("s3", region_name=region)
        response = client.list_buckets()
        buckets = response.get("Buckets", [])
        if not buckets:
            return "No S3 buckets found in this account."
        lines = ["# S3 buckets\n"]
        for b in buckets:
            name = b.get("Name", "?")
            created = b.get("CreationDate", "")
            lines.append(f"- {name} (created: {created})")
        return "\n".join(lines)
    except (BotoCoreError, ClientError) as e:
        logger.warning("fetch_s3_summary failed: %s", e)
        return f"Error fetching S3 buckets: {e}"


def fetch_cloudwatch_alarms_summary(region: str) -> str:
    try:
        client = boto3.client("cloudwatch", region_name=region)
        response = client.describe_alarms(MaxRecords=100)
        alarms = response.get("MetricAlarms", []) + response.get("CompositeAlarms", [])
        if not alarms:
            return "No CloudWatch alarms found in this account/region."
        lines = ["# CloudWatch alarms\n"]
        for a in alarms:
            name = a.get("AlarmName", "?")
            desc = a.get("AlarmDescription") or "(no description)"
            state = a.get("StateValue", "?")
            lines.append(f"- {name}: state={state}\n  {desc}")
        return "\n".join(lines)
    except (BotoCoreError, ClientError) as e:
        logger.warning("fetch_cloudwatch_alarms_summary failed: %s", e)
        return f"Error fetching CloudWatch alarms: {e}"


SOURCES = {
    "ecs": fetch_ecs_summary,
    "s3": fetch_s3_summary,
    "cloudwatch": fetch_cloudwatch_alarms_summary,
}


def fetch_source(source: str, region: str) -> str:
    if source not in SOURCES:
        raise ValueError(f"Unknown source: {source}. Choose from: {list(SOURCES)}")
    return SOURCES[source](region)
