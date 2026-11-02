from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ..documents import Document
from .base import content_type_for, title_from


class S3Connector:
    """Reads supported objects under `s3://bucket/prefix`. Credentials come from the standard
    AWS chain (env, profile, IRSA), never from the tenant YAML."""

    def __init__(self, name: str, bucket: str, prefix: str = "", client: Any = None, **kw) -> None:
        self.name = name
        self.bucket = bucket
        self.prefix = prefix
        if client is None:
            import boto3  # optional dependency: pip install enterprise-rag-platform[s3]

            client = boto3.client("s3", region_name=kw.get("region"))
        self.client = client

    def fetch(self) -> Iterator[Document]:
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                ct = content_type_for(key)
                if not ct:
                    continue
                body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
                text = body.decode("utf-8", errors="replace")
                yield Document(
                    doc_id=f"{self.name}:{key}",
                    source=self.name,
                    title=title_from(key, text, ct),
                    content=text,
                    content_type=ct,
                    metadata={"s3_key": key, "etag": obj.get("ETag", "").strip('"')},
                )
