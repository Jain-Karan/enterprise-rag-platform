"""Generate the synthetic enterprise corpus and golden question sets used by the benchmark.

Why synthetic: real enterprise policy docs can't be published. The generator reproduces the
property that makes them hard to retrieve over: many documents share the same structure
(every line of business has a "Settlement Authority" table with the same row labels), so the
row "| Senior Adjuster | $41,250 |" is meaningless without its document title and section.

Every answer value is unique across the corpus, so "the retrieved chunk contains the answer
string" is an unambiguous hit signal.

    python benchmarks/generate_corpus.py            # writes benchmarks/data/
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).parent / "data"

FILLER = [
    "This standard applies to all employees and contractors acting on behalf of the company.",
    "Exceptions must be documented and approved before the action is taken, not after.",
    "Owners review this document annually or after any material regulatory change.",
    "Where this document conflicts with local law, local law takes precedence.",
    "Questions about interpretation should be raised with the document owner.",
    "Records created under this standard are retained according to the retention schedule.",
    "Audit findings related to this standard are tracked to closure by the control owner.",
    "Teams may adopt stricter thresholds but may not relax the ones defined here.",
]

# Boilerplate sections every real policy carries. They add length and shared vocabulary
# ("limit", "approval", "threshold") without carrying any golden answers.
GENERIC_SECTIONS = {
    "Definitions": [
        "- Approver: the person accountable for authorising an action under this standard.",
        "- Threshold: a limit above which additional review or approval is required.",
        "- Exception: a documented, time-bound deviation approved by the document owner.",
        "- Control owner: the role responsible for operating and evidencing a control.",
    ],
    "Roles and Responsibilities": [
        "- The document owner maintains this standard and approves exceptions.",
        "- Line managers ensure their teams complete required training each year.",
        "- Internal audit tests the controls described here on a risk-based cycle.",
        "- Compliance monitors adherence and reports breaches to the risk committee.",
    ],
    "Training": [
        "- New joiners complete the relevant training module before acting under this standard.",
        "- Refresher training is required annually and when limits or thresholds change.",
    ],
    "Breach Reporting": [
        "- Suspected breaches of a limit are reported to compliance on the same business day.",
        "- Repeated breaches by the same team trigger a review of delegated authority.",
    ],
    "Related Documents": [
        "- Enterprise risk management framework.",
        "- Delegated authority register.",
        "- Records retention schedule.",
    ],
}

# (section title, kind, spec). kind: table | facts
TENANTS = {
    "acme-insurance": {
        "topics": {
            "Claims Handling Standard": {
                "variants": [
                    "Auto",
                    "Homeowners",
                    "Commercial Property",
                    "Marine Cargo",
                    "Workers Compensation",
                    "Professional Liability",
                    "Cyber",
                    "Directors and Officers",
                    "Product Liability",
                    "Inland Marine",
                    "Umbrella",
                    "Crop",
                ],
                "sections": [
                    (
                        "Settlement Authority",
                        "table",
                        {
                            "cols": ["Role", "Maximum settlement"],
                            "rows": [
                                "Adjuster I",
                                "Adjuster II",
                                "Senior Adjuster",
                                "Claims Manager",
                                "Regional Director",
                                "Chief Claims Officer",
                            ],
                            "fmt": "${:,}",
                            "range": (5_000, 900_000),
                            "q": [
                                "What is the maximum settlement a {row} can approve for {variant} claims?",
                                "{variant} claims: settlement authority limit for a {row}?",
                            ],
                        },
                    ),
                    (
                        "Service Levels",
                        "table",
                        {
                            "cols": ["Step", "Target"],
                            "rows": [
                                "First notice acknowledgement",
                                "Initial customer contact",
                                "Inspection scheduled",
                                "Coverage decision",
                                "Payment issued",
                                "Subrogation review",
                            ],
                            "fmt": "{} business hours",
                            "range": (2, 400),
                            "q": [
                                "What is the target time for {row} on a {variant} claim?",
                                "For {variant} claims, how fast must {row} happen?",
                            ],
                        },
                    ),
                    (
                        "Fraud Referral",
                        "facts",
                        [
                            (
                                "{variant} claims with a fraud model score above {v} are referred to the "
                                "Special Investigations Unit.",
                                "{}",
                                (500, 990),
                                "Above what fraud score are {variant} claims sent to the Special Investigations Unit?",
                            ),
                            (
                                "Referred {variant} files must be acknowledged by SIU within {v}.",
                                "{} minutes",
                                (10, 900),
                                "How quickly must SIU acknowledge a referred {variant} claim?",
                            ),
                        ],
                    ),
                ],
            },
            "Underwriting Guidelines": {
                "variants": [
                    "Small Business",
                    "Mid Market",
                    "Habitational",
                    "Construction",
                    "Hospitality",
                    "Manufacturing",
                    "Healthcare",
                    "Technology",
                    "Retail",
                    "Transportation",
                ],
                "sections": [
                    (
                        "Referral Thresholds",
                        "table",
                        {
                            "cols": ["Exposure", "Refer above"],
                            "rows": [
                                "Total insured value",
                                "Single location value",
                                "Annual revenue",
                                "Prior losses (5 yr)",
                                "Vacancy period",
                            ],
                            "fmt": "${:,}",
                            "range": (10_000, 9_000_000),
                            "q": [
                                "In {variant} underwriting, above what {row} must a risk be referred?",
                                "Referral threshold for {row} under the {variant} guidelines?",
                            ],
                        },
                    ),
                    (
                        "Pricing Guardrails",
                        "facts",
                        [
                            (
                                "The maximum schedule rating credit for {variant} accounts is {v}.",
                                "{} basis points",
                                (100, 2500),
                                "What is the largest schedule rating credit allowed for {variant} accounts?",
                            ),
                            (
                                "Quotes for {variant} risks expire after {v}.",
                                "{} calendar days",
                                (15, 400),
                                "When does a {variant} quote expire?",
                            ),
                        ],
                    ),
                ],
            },
            "Travel and Expense Policy": {
                "variants": [
                    "United States",
                    "Canada",
                    "United Kingdom",
                    "India",
                    "Singapore",
                    "Germany",
                    "Japan",
                    "Brazil",
                    "Australia",
                    "France",
                    "Mexico",
                    "United Arab Emirates",
                ],
                "sections": [
                    (
                        "Daily Limits",
                        "table",
                        {
                            "cols": ["Category", "Daily limit"],
                            "rows": [
                                "Lodging",
                                "Meals",
                                "Ground transport",
                                "Client entertainment",
                                "Incidentals",
                            ],
                            "fmt": "{:,} local currency units",
                            "range": (20, 9_000),
                            "q": [
                                "What is the daily {row} limit for travel in {variant}?",
                                "{variant} travel: how much can I spend per day on {row}?",
                            ],
                        },
                    ),
                    (
                        "Approvals",
                        "facts",
                        [
                            (
                                "Trips to {variant} longer than {v} need VP approval.",
                                "{} nights",
                                (3, 300),
                                "How long can a {variant} trip be before VP approval is needed?",
                            ),
                        ],
                    ),
                ],
            },
        },
        "sources": [
            ("sharepoint-policies", "md"),
            ("confluence-wiki", "html"),
            ("s3-exports", "md"),
        ],
    },
    "northwind-bank": {
        "topics": {
            "Payments Operations Runbook": {
                "variants": [
                    "ACH",
                    "Domestic Wire",
                    "International Wire",
                    "RTP",
                    "FedNow",
                    "Card Settlement",
                    "SWIFT gpi",
                    "SEPA",
                    "Check Clearing",
                ],
                "sections": [
                    (
                        "Cutoff and Limits",
                        "table",
                        {
                            "cols": ["Control", "Value"],
                            "rows": [
                                "Single payment limit",
                                "Daily customer limit",
                                "Dual control threshold",
                                "Manual review threshold",
                            ],
                            "fmt": "${:,}",
                            "range": (1_000, 50_000_000),
                            "q": ["What is the {row} for {variant} payments?", "{variant}: {row}?"],
                        },
                    ),
                    (
                        "Exception Handling",
                        "facts",
                        [
                            (
                                "Unmatched {variant} returns are escalated after {v}.",
                                "{} minutes",
                                (5, 999),
                                "When are unmatched {variant} returns escalated?",
                            ),
                        ],
                    ),
                ],
            },
            "Credit Risk Policy": {
                "variants": [
                    "Commercial Real Estate",
                    "SME Lending",
                    "Consumer Cards",
                    "Auto Loans",
                    "Mortgages",
                    "Equipment Finance",
                    "Agricultural Lending",
                    "Leveraged Finance",
                ],
                "sections": [
                    (
                        "Approval Matrix",
                        "table",
                        {
                            "cols": ["Approver", "Exposure limit"],
                            "rows": [
                                "Relationship Manager",
                                "Credit Officer",
                                "Senior Credit Officer",
                                "Credit Committee",
                            ],
                            "fmt": "${:,}",
                            "range": (10_000, 90_000_000),
                            "q": [
                                "Up to what exposure can a {row} approve in {variant}?",
                                "{variant} approval matrix: {row} limit?",
                            ],
                        },
                    ),
                ],
            },
        },
        "sources": [("sharepoint-runbooks", "md"), ("confluence-risk", "html")],
    },
}


class UniquePool:
    """Hands out numbers whose formatted answer strings never repeat anywhere in the corpus."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.used: set[str] = set()

    def draw(self, lo: int, hi: int, fmt: str) -> tuple[int, str]:
        for _ in range(10_000):
            v = self.rng.randint(lo, hi)
            if lo >= 1000:
                v = v // 10 * 10 + self.rng.choice([0, 5])
            s = fmt.format(v)
            num = s.split()[0]
            if num not in self.used:
                self.used.add(num)
                return v, s
        raise RuntimeError("value pool exhausted")


def render_md(title: str, sections: list[tuple[str, list]]) -> str:
    out = [f"# {title}", ""]
    for heading, blocks in sections:
        out += [f"## {heading}", ""]
        for kind, payload in blocks:
            if kind == "p":
                out += [payload, ""]
            elif kind == "list":
                out += payload + [""]
            elif kind == "table":
                cols, rows = payload
                out.append("| " + " | ".join(cols) + " |")
                out.append("|" + "---|" * len(cols))
                out += ["| " + " | ".join(r) + " |" for r in rows]
                out.append("")
    return "\n".join(out)


def render_html(title: str, sections: list[tuple[str, list]]) -> str:
    out = [f"<html><head><title>{title}</title></head><body>", f"<h1>{title}</h1>"]
    for heading, blocks in sections:
        out.append(f"<h2>{heading}</h2>")
        for kind, payload in blocks:
            if kind == "p":
                out.append(f"<p>{payload}</p>")
            elif kind == "list":
                out.append("<ul>" + "".join(f"<li>{i[2:]}</li>" for i in payload) + "</ul>")
            elif kind == "table":
                cols, rows = payload
                out.append("<table><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>")
                out += ["<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows]
                out.append("</table>")
    out.append("</body></html>")
    return "\n".join(out)


def build(seed: int = 7) -> dict[str, int]:
    rng = random.Random(seed)
    pool = UniquePool(rng)
    if ROOT.exists():
        shutil.rmtree(ROOT)
    counts = {}
    for tenant, spec in TENANTS.items():
        golden: list[dict] = []
        docs = [(topic, v, t) for topic, t in spec["topics"].items() for v in t["variants"]]
        rng.shuffle(docs)
        for i, (topic, variant, tspec) in enumerate(docs):
            src, fmt = spec["sources"][i % len(spec["sources"])]
            title = f"{topic}: {variant}"
            slug = f"{topic}-{variant}".lower().replace(" ", "-").replace(":", "")
            doc_id = f"{src}:{slug}.{fmt}"
            sections: list[tuple[str, list]] = [
                ("Purpose and Scope", [("p", " ".join(rng.sample(FILLER, 3)))])
            ]
            for name in rng.sample(sorted(GENERIC_SECTIONS), 2):
                sections.append(
                    (name, [("p", rng.choice(FILLER)), ("list", GENERIC_SECTIONS[name])])
                )
            for heading, kind, s in tspec["sections"]:
                blocks: list = [("p", " ".join(rng.sample(FILLER, rng.randint(2, 4))))]
                if kind == "table":
                    rows = []
                    # authority limits and SLAs grow down the table, as they do in real docs
                    drawn = sorted(pool.draw(*s["range"], s["fmt"]) for _ in s["rows"])
                    for row, (_, ans) in zip(s["rows"], drawn, strict=True):
                        rows.append([row, ans])
                        q = rng.choice(s["q"]).format(row=row.lower(), variant=variant)
                        golden.append(
                            {
                                "id": f"{tenant}-{len(golden)}",
                                "question": q,
                                "answer": ans,
                                "doc_id": doc_id,
                            }
                        )
                    blocks.append(("table", (s["cols"], rows)))
                else:
                    for sentence, afmt, rng_, q in s:
                        _, ans = pool.draw(*rng_, afmt)
                        blocks.append(("p", sentence.format(variant=variant, v=ans)))
                        golden.append(
                            {
                                "id": f"{tenant}-{len(golden)}",
                                "question": q.format(variant=variant),
                                "answer": ans,
                                "doc_id": doc_id,
                            }
                        )
                for _ in range(rng.randint(1, 3)):
                    blocks.append(("p", " ".join(rng.sample(FILLER, rng.randint(2, 4)))))
                sections.append((heading, blocks))
            for name in rng.sample(sorted(GENERIC_SECTIONS), 2):
                sections.append(
                    (name, [("p", rng.choice(FILLER)), ("list", GENERIC_SECTIONS[name])])
                )
            sections.append(("Document Control", [("p", " ".join(rng.sample(FILLER, 2)))]))
            body = render_md(title, sections) if fmt == "md" else render_html(title, sections)
            path = ROOT / tenant / src / f"{slug}.{fmt}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body)
        (ROOT / tenant / "golden.jsonl").write_text("\n".join(json.dumps(g) for g in golden) + "\n")
        counts[tenant] = len(golden)
    return counts


if __name__ == "__main__":
    for tenant, n in build().items():
        print(f"{tenant}: {n} golden questions")
