"""`ragp` command line.

ragp onboard  -c config/tenants.yaml -t acme     # ingest every source, then run the golden set
ragp ingest   -c config/tenants.yaml -t acme
ragp query    -c config/tenants.yaml -t acme "What is the per diem for Tokyo?"
ragp serve    -c config/tenants.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from .bench import evaluate, load_golden
from .config import load_config
from .platform import Platform


def _platform(args: argparse.Namespace) -> Platform:
    return Platform(load_config(args.config))


def cmd_ingest(args: argparse.Namespace) -> int:
    r = _platform(args).ingest(args.tenant)
    print(
        json.dumps(
            {
                "documents": r.documents,
                "chunks": r.chunks,
                "per_source": r.per_source,
                "seconds": round(r.seconds, 2),
            },
            indent=2,
        )
    )
    return 0


def cmd_onboard(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    p = _platform(args)
    tenant = p.config.tenant(args.tenant)
    r = p.ingest(args.tenant)
    out: dict = {
        "tenant_id": args.tenant,
        "documents": r.documents,
        "chunks": r.chunks,
        "per_source": r.per_source,
    }
    if tenant.benchmark:
        res = evaluate(
            lambda q: p.retrieve(args.tenant, q),
            load_golden(tenant.benchmark),
            k=tenant.retrieval.top_k,
        )
        out["benchmark"] = {k: v for k, v in res.as_dict().items() if k != "misses"}
    else:
        out["benchmark"] = "skipped: no golden set configured for this tenant"
    out["wall_seconds"] = round(time.perf_counter() - t0, 2)
    print(json.dumps(out, indent=2))
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    a = _platform(args).answer(args.tenant, args.question)
    print(a.answer)
    for c in a.citations:
        print(f"  [{c['n']}] {c['title']} > {c['section']}  ({c['doc_id']})")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(_platform(args)), host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ragp")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser, tenant: bool = True) -> None:
        p.add_argument("-c", "--config", default="config/tenants.yaml")
        if tenant:
            p.add_argument("-t", "--tenant", required=True)

    p = sub.add_parser("ingest")
    common(p)
    p.set_defaults(fn=cmd_ingest)
    p = sub.add_parser("onboard")
    common(p)
    p.set_defaults(fn=cmd_onboard)
    p = sub.add_parser("query")
    common(p)
    p.add_argument("question")
    p.set_defaults(fn=cmd_query)
    p = sub.add_parser("serve")
    common(p, tenant=False)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_serve)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
