import os
import json
import boto3
from typing import List

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name=os.environ["AWS_REGION"])
    return _client


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of text chunks using Amazon Titan Embed Text v1 via Bedrock.
    Returns a list of embedding vectors (one per chunk).
    """
    model_id = os.environ.get("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v1")
    client = _get_client()
    embeddings = []

    for text in texts:
        body = json.dumps({"inputText": text})
        response = client.invoke_model(
            modelId=model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        embeddings.append(result["embedding"])

    return embeddings
